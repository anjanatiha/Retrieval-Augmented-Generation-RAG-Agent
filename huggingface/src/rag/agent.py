"""agent.py — Agent class. HF Space version.

Replaces all ollama.chat() calls with self.store._llm_chat().
Everything else is identical to the local version.
"""

import re

from src.rag.vector_store import VectorStore

__all__ = ['Agent']


class Agent:
    AGENT_SYSTEM_PROMPT = """You are an AI agent. You must ONLY respond with tool calls — no explanations, no extra text.

Available tools:
1. rag_search - search the knowledge base for information
2. calculator - evaluate a math expression
3. summarise  - summarise a piece of text
4. sentiment  - analyse the sentiment/tone of a passage or topic from the documents
5. finish     - return the final answer to the user

You MUST respond in EXACTLY this format with NO other text before or after:
TOOL: tool_name(your argument here)

Examples:
TOOL: rag_search(NLP experience)
TOOL: calculator(16 * 365)
TOOL: summarise(cats sleep a lot and are nocturnal hunters...)
TOOL: sentiment(customer reviews)
TOOL: finish(Yes, the candidate has NLP experience including POS tagging and language modeling.)

Rules:
- Never write anything except a single TOOL: line
- Always end with TOOL: finish(your final answer)
- Use rag_search first to find information before answering
- Do not explain yourself or add any commentary
- The finish argument must be a clean, direct answer in plain English — NEVER paste raw bullet points or document chunks into finish
- rag_search arguments must be SHORT, SIMPLE keyword phrases — NEVER use boolean operators like AND, OR, quotes, or complex syntax
- For simple math questions, call calculator once then finish
- For simple factual questions, call rag_search once then finish
- For sentiment questions (e.g. "what is the tone", "is this positive", "sentiment of"), call sentiment with a keyword or passage then finish
- For summarisation or comprehensive tasks (e.g. "summarise", "tell me about", "what is in"):
  * Make multiple SEPARATE rag_search calls, one per topic
  * For a resume: search "work experience", then "education", then "skills", then "projects" as separate calls
  * Collect all results, then call finish with a complete summary
"""

    def __init__(self, store: VectorStore):
        self.store             = store
        self.messages          = []
        self.collected_context = []
        self.max_steps         = 8

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, user_query, streamlit_mode=False):
        """Run the ReAct agent loop for a user query."""
        self.messages = [
            {'role': 'system', 'content': self.AGENT_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_query},
        ]
        self.collected_context = []

        steps            = []
        answer           = None
        bad_format_count = 0

        _q_lower = user_query.lower()
        is_summarise = (
            any(s in _q_lower for s in
                ['summarise', 'summarize', 'summerise', 'summerize', 'summary',
                 'overview', 'tell me about', 'describe', 'what is in'])
            or _q_lower.startswith('summ')
        )

        if is_summarise:
            return self._fast_path_summarise(user_query, streamlit_mode)

        # Fast path: pure math — skip LLM entirely, evaluate directly
        if self._is_math_expression(user_query):
            return self._fast_path_calculator(user_query)

        is_sentiment = any(s in _q_lower for s in
                           ['sentiment', 'tone', 'feeling', 'positive', 'negative', 'neutral',
                            'emotion', 'attitude', 'mood'])
        if is_sentiment:
            return self._fast_path_sentiment(user_query, streamlit_mode)

        for step in range(self.max_steps):
            raw_text = self.store._llm_chat(self.messages)
            tool_name, tool_arg = self._parse_tool_call(raw_text)

            if not tool_name:
                bad_format_count += 1
                if bad_format_count <= 2:
                    self.messages.append({'role': 'assistant', 'content': raw_text})
                    self.messages.append({'role': 'user', 'content':
                        'Wrong format. You must respond with ONLY this format — nothing else:\n'
                        'TOOL: tool_name(argument)\n'
                        'Example: TOOL: rag_search(cat sleep hours)'})
                    continue
                else:
                    answer = raw_text
                    steps.append({'step': step+1, 'tool': 'none', 'arg': '', 'result': raw_text})
                    break

            bad_format_count = 0

            if tool_name == 'finish':
                if self.collected_context:
                    all_context = '\n'.join(self.collected_context)
                    answer = self._synthesize_final_answer(user_query, all_context)
                else:
                    answer = tool_arg
                steps.append({'step': step+1, 'tool': 'finish', 'arg': answer, 'result': answer})
                break

            result = self._dispatch_tool(tool_name, tool_arg)
            steps.append({'step': step+1, 'tool': tool_name, 'arg': tool_arg, 'result': result})

            if tool_name == 'calculator' and not result.startswith('Error'):
                answer = f"{tool_arg} = {result}"
                steps.append({'step': step+2, 'tool': 'finish', 'arg': answer, 'result': answer})
                break

            if tool_name == 'rag_search' and not is_summarise:
                answer = self._synthesize_final_answer(user_query, result)
                steps.append({'step': step+2, 'tool': 'finish', 'arg': answer, 'result': answer})
                break

            self.messages.append({'role': 'assistant', 'content': raw_text})
            self.messages.append({'role': 'user', 'content':
                f"Tool result: {result}\n\n"
                f"Original task: {user_query}\n\n"
                f"If you now have enough information to answer the original task, call:\n"
                f"TOOL: finish(your answer)\n\n"
                f"Otherwise call the next tool. Respond ONLY with a single TOOL: line."})

        if answer is None:
            answer = "Agent reached max steps without a final answer."

        return {'answer': answer, 'steps': steps}

    # ── Private — loop ───────────────────────────────────────────────────────

    def _parse_tool_call(self, response_text):
        # Greedy (.+) + no DOTALL: captures up to the LAST ')' on the line,
        # so nested parens in expressions like "7+(9+8)-2*6" are preserved.
        match = re.search(r'(?i)TOOL:\s*(\w+)\s*\(\s*(.+)\s*\)', response_text)
        if match:
            return match.group(1).strip().lower(), match.group(2).strip()
        match = re.search(r'(?i)TOOL:\s*(\w+)\s+(.+)', response_text)
        if match:
            return match.group(1).strip().lower(), match.group(2).strip()
        return None, None

    def _dispatch_tool(self, tool_name, tool_arg):
        if tool_name == 'rag_search':
            result = self._tool_rag_search(tool_arg)
            self.collected_context.append(f"[Search: {tool_arg}]\n{result}")
        elif tool_name == 'calculator':
            result = self._tool_calculator(tool_arg)
        elif tool_name == 'summarise':
            result = self._tool_summarise(tool_arg)
        elif tool_name == 'sentiment':
            result = self._tool_sentiment(tool_arg)
            self.collected_context.append(f"[Sentiment analysis: {tool_arg}]\n{result}")
        else:
            result = (f"Unknown tool '{tool_name}'. "
                      f"Available: rag_search, calculator, summarise, sentiment, finish")
        return result

    def _synthesize_final_answer(self, query, context):
        prompt = (
            "You are a helpful assistant. Answer the question below using ONLY the "
            "provided context. Be concise and direct. Do not repeat the context — "
            "just answer the question. Cite the source filename at the end.\n"
            "Stop after answering — do NOT add more examples, more questions, or more context blocks.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        try:
            raw = self.store._llm_chat([{'role': 'user', 'content': prompt}], temperature=0)
            # Truncate any hallucinated follow-up Q&A blocks the LLM appends
            raw = re.split(r'\n\n(?:Context:|Question:)', raw, maxsplit=1)[0]
            return raw.strip()
        except Exception as e:
            # Return a clean error rather than dumping raw chunk text
            return f"(Could not synthesize answer: {e})"

    def _fast_path_summarise(self, query, streamlit_mode=False):
        """For summarise queries: extract topic from query, run targeted searches."""
        # Strip summarise keywords to get the actual topic
        topic = re.sub(
            r'\b(summarise|summarize|summerise|summerize|summary|overview|'
            r'tell me about|describe|what is in|give me a|the|a|an|document|doc|file)\b',
            '', query, flags=re.IGNORECASE
        ).strip() or query

        # Use resume-specific terms if query is about a resume/CV
        _resume_keywords = ['resume', 'cv', 'curriculum vitae', 'candidate', 'applicant']
        if any(k in query.lower() for k in _resume_keywords):
            search_terms = ['work experience', 'education', 'skills projects', 'summary contact']
        else:
            search_terms = [topic, f"{topic} overview", f"{topic} details", f"{topic} facts"]

        fast_steps = []
        collected  = []
        for i, term in enumerate(search_terms):
            r = self._tool_rag_search(term)
            collected.append(f"[Search: {term}]\n{r}")
            fast_steps.append({'step': i+1, 'tool': 'rag_search', 'arg': term, 'result': r})
        all_context = '\n\n'.join(collected)
        answer = self._synthesize_final_answer(query, all_context)
        fast_steps.append({'step': len(search_terms)+1, 'tool': 'finish',
                           'arg': answer, 'result': answer})
        return {'answer': answer, 'steps': fast_steps}

    def _fast_path_sentiment(self, query, streamlit_mode=False):
        """For sentiment queries: search then analyse directly."""
        _q_lower = query.lower()
        search_query = re.sub(
            r'\b(what is the|what\'s the|analyse|analyze|analysis|check|tell me the|'
            r'of the|of|sentiment|tone|feeling|emotion|attitude|mood|the|is|a|an)\b',
            '', _q_lower, flags=re.IGNORECASE
        ).strip() or query
        raw = self._tool_rag_search(search_query)
        clean_text = re.sub(r'-\s*\[[^\]]+\]\s*', '', raw).strip()
        sentiment_result = self._tool_sentiment(clean_text or raw)
        sentiment_steps = [
            {'step': 1, 'tool': 'rag_search',  'arg': search_query,     'result': raw},
            {'step': 2, 'tool': 'sentiment',    'arg': search_query,     'result': sentiment_result},
            {'step': 3, 'tool': 'finish',       'arg': sentiment_result, 'result': sentiment_result},
        ]
        return {'answer': sentiment_result, 'steps': sentiment_steps}

    # ── Private — tools ──────────────────────────────────────────────────────

    def _is_math_expression(self, text):
        """Return True if the query is a pure math expression with no words."""
        # Strip spaces and check only math chars remain (digits, operators, parens, dot, %)
        stripped = re.sub(r'\s+', '', text)
        return bool(re.fullmatch(r'[\d\+\-\*\/\(\)\.\%]+', stripped))

    def _fast_path_calculator(self, user_query):
        """Evaluate a pure math expression directly — no LLM call."""
        result = self._tool_calculator(user_query.strip())
        answer = f"{user_query.strip()} = {result}"
        steps  = [
            {'step': 1, 'tool': 'calculator', 'arg': user_query.strip(), 'result': result},
            {'step': 2, 'tool': 'finish',     'arg': answer,             'result': answer},
        ]
        return {'answer': answer, 'steps': steps}

    def _tool_rag_search(self, query):
        """Returns retrieved chunks with source labels for grounded synthesis."""
        queries   = self.store._expand_query(query)
        retrieved = self.store._hybrid_retrieve(queries, top_n=5)
        reranked  = self.store._rerank(query, retrieved, top_n=3)
        lines = []
        for e, sim, _ in reranked:
            label = self.store._source_label(e)
            lines.append(f"- [{e['source']} {label}] {e['text']}")
        return '\n'.join(lines)

    def _tool_calculator(self, expression):
        try:
            # Normalise percentage expressions before safety check
            # "15% of 85000" → "(15/100*85000)"
            # "15%" alone → "(15/100)"
            expr = re.sub(
                r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)',
                r'(\1/100*\2)',
                expression,
                flags=re.IGNORECASE,
            )
            expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
            allowed = set('0123456789+-*/(). ')
            if not all(c in allowed for c in expr):
                return "Error: unsafe expression"
            return str(eval(expr))
        except Exception as e:
            return f"Error: {e}"

    def _tool_summarise(self, text):
        word_count = len(text.split())
        if word_count < 100:
            length_hint = "2-3 sentences"
        elif word_count < 300:
            length_hint = "4-5 sentences"
        else:
            length_hint = "6-8 sentences covering all key points"
        return self.store._llm_chat(
            [{'role': 'user', 'content': f"Summarise this in {length_hint}:\n{text}"}]
        )

    def _tool_sentiment(self, text_or_query):
        """Analyses sentiment/tone. Short queries search first; longer text analyses directly."""
        if len(text_or_query.split()) < 10:
            retrieved = self._tool_rag_search(text_or_query)
            text_to_analyse = retrieved if retrieved.strip() else text_or_query
        else:
            text_to_analyse = text_or_query

        prompt = (
            "Analyse the sentiment and tone of the following text.\n\n"
            "Respond ONLY in this exact format — nothing else, no extra text, no examples:\n"
            "Sentiment: <Positive / Negative / Neutral / Mixed>\n"
            "Tone: <one short phrase describing the tone, e.g. 'professional and confident'>\n"
            "Key phrases: <2-4 phrases from the text that drove this assessment>\n"
            "Explanation: <1-2 sentences explaining the sentiment>\n\n"
            "Do NOT add any additional text, examples, or commentary after the Explanation line.\n\n"
            f"Text:\n{text_to_analyse}"
        )
        try:
            return self.store._llm_chat(
                [{'role': 'user', 'content': prompt}], temperature=0
            )
        except Exception as e:
            return f"Sentiment analysis error: {e}"
