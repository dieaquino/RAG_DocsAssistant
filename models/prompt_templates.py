"""
Prompt templates for different types of insurance queries with adaptive tone
"""
from typing import Dict, Any, List

class PromptTemplates:
    """Manages prompt templates for different query types and response modes"""

    def __init__(self):
        ## IMPROVEMENT: The base prompt is now stricter and more explicit.
        self.base_system_prompt = """You are a world-class specialized assistant for Zurich Insurance policies.
Your primary directive is to provide precise, factual, and helpful answers to users about their life insurance coverage.

**CRITICAL INSTRUCTIONS - FOLLOW THESE AT ALL TIMES:**
1.  **Grounding:** Your entire response MUST be based EXCLUSIVELY on the information provided in the 'POLICY INFORMATION' context. Do not use any prior knowledge.
2.  **No Hallucination:** If the answer is not in the provided context, you MUST state: "I could not find specific information about this topic in the policy." Do NOT invent, infer, or guess information.
3.  **Citation is Mandatory:** For every piece of information you provide, you MUST cite the source (e.g., `[page 5]`, `[Section: Eligibility, page 12]`). If a page is not available, cite the section.
4.  **Precision:** Quote specific amounts, dates, percentages, and deadlines exactly as they appear in the context.
5.  **Self-Correction:** Before generating the final answer, internally review your response to ensure it strictly adheres to all these instructions.
"""

    def get_system_prompt(self, query_type: str, response_mode: str) -> str:
        """Get system prompt adapted for query type and response mode"""

        ## IMPROVEMENT: Tone adaptations are now more action-oriented.
        tone_adaptations = {
            "benefits": "Your goal is to clarify value. Be meticulous with numbers and conditions. Structure the answer to be easily scannable.",
            "definitions": "Your goal is to provide clarity. Act like a teacher. Explain complex terms simply after quoting them.",
            "claims": "Your goal is to guide the user. Use a professional, calm, and empathetic tone. Provide a step-by-step checklist.",
            "eligibility": "Your goal is to verify requirements. Be direct and factual. Use a checklist format to compare policy rules with user data if available.",
            "policy_admin": "Your goal is to explain processes. Be formal and procedural. Clearly distinguish between employee and employer responsibilities.",
            "general": "Your goal is to be a helpful navigator. Be conversational but always ground your answers in the provided text and suggest more specific queries."
        }

        mode_instructions = {
            "concise": "Provide a direct, summary-level answer. Get straight to the point.",
            "detailed": "Provide a comprehensive answer. Include explanations, examples, and related details found in the context.",
            "custom": "Adapt the level of detail based on the user's question. A simple question gets a simple answer; a complex one gets a more detailed breakdown."
        }

        adaptation = tone_adaptations.get(query_type, tone_adaptations["general"])
        mode_instruction = mode_instructions.get(response_mode, mode_instructions["custom"])

        return f"{self.base_system_prompt}\n\n**TASK-SPECIFIC DIRECTIVE:** {adaptation}\n\n**RESPONSE DETAIL LEVEL:** {mode_instruction}"

    def get_user_prompt(
        self,
        query: str,
        context_data: Dict[str, Any],
        response_mode: str,
        user_data: Dict[str, Any] = None
    ) -> str:
        """Generate user prompt with context and query"""
        contexts = context_data.get('contexts', [])
        query_type = context_data.get('query_type', 'general')
        confidence = context_data.get('confidence', 0.0)

        context_text = self._build_context_section(contexts)
        user_section = self._build_user_section(user_data) if user_data else ""
        metadata_section = self._build_metadata_section(context_data)
        template = self._get_query_template(query_type, response_mode)

        return template.format(
            query=query,
            context=context_text,
            user_info=user_section,
            metadata=metadata_section,
            confidence_level=f"{confidence*100:.1f}%"
        )

    def _build_context_section(self, contexts: List[Dict[str, Any]]) -> str:
        """Build formatted context section from retrieved chunks"""
        if not contexts:
            return "No specific information was found in the policy for this query."

        context_parts = [
            "## NOTE: Initial instruction for the LLM.\n"
            "The following are excerpts from the user's insurance policy. This is the ONLY source of truth you can use."
        ]
        for i, ctx in enumerate(contexts, 1):
            content = ctx['content']
            metadata = ctx.get('metadata', {})
            page = metadata.get('page', 'N/A')
            chunk_type = metadata.get('chunk_type', 'general')
            context_part = f"""\n--- Source {i} (Page: {page}, Section: {chunk_type}) ---\n{content}"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _build_user_section(self, user_data: Dict[str, Any]) -> str:
        """Build user information section"""
        if not user_data:
            return ""
        user_info_parts = [f"{key.replace('_', ' ').title()}: {value}" for key, value in user_data.items() if value]
        if user_info_parts:
            return "\n--- USER INFORMATION (for context) ---\n" + "\n".join(user_info_parts) + "\n"
        return ""

    def _build_metadata_section(self, context_data: Dict[str, Any]) -> str:
        """Build metadata section for internal reference"""
        query_type = context_data.get('query_type', 'general')
        total_chunks = context_data.get('total_chunks', 0)
        sources = context_data.get('sources', [])
        return f"""--- INTERNAL METADATA ---\nQuery Type: {query_type}\nChunks Analyzed: {total_chunks}\nSources: {', '.join(sources) if sources else 'N/A'}"""

    def _get_query_template(self, query_type: str, response_mode: str) -> str:
        """Get specific template based on query type"""
        templates = {
            "benefits": self._get_benefits_template(response_mode),
            "definitions": self._get_definitions_template(response_mode),
            "claims": self._get_claims_template(response_mode),
            "eligibility": self._get_eligibility_template(response_mode),
            "policy_admin": self._get_admin_template(response_mode),
            "general": self._get_general_template(response_mode)
        }
        return templates.get(query_type, templates["general"])

    ## IMPROVEMENT: All templates now include a "Thought Process" section to guide the LLM.
    def _get_benefits_template(self, response_mode: str) -> str:
        return """User's Question: {query}

POLICY INFORMATION:
{context}
{user_info}

INSTRUCTIONS:
1.  **Thought Process:**
    a. Identify the specific benefit in the user's query.
    b. Scan the context for all monetary amounts, percentages, and conditions related to this benefit.
    c. If user information (like age) is available, check for clauses on how it affects the benefit.
    d. Structure the answer logically before writing.
2.  **Answer Generation:**
    a. Start with the primary coverage amount.
    b. List any conditions or requirements for the benefit to be paid.
    c. Detail any reductions (e.g., age-based) or exclusions.
    d. Cite the source for every piece of data (e.g., `[page 10]`).

METADATA: {metadata}
Response Confidence: {confidence_level}

Answer the user's question:"""

    def _get_definitions_template(self, response_mode: str) -> str:
        return """User's Question: {query}

DEFINITIONS IN THE POLICY:
{context}
{user_info}

INSTRUCTIONS:
1.  **Thought Process:**
    a. Locate the exact definition for the term in the user's query.
    b. Analyze the definition: is it complex? Does it reference other terms?
    c. Think of a simple analogy or example to clarify it.
2.  **Answer Generation:**
    a. Quote the definition verbatim from the policy, followed by its source citation.
    b. In a new paragraph, explain the definition in simple, clear language.
    c. Provide a practical example of how this term applies.

METADATA: {metadata}
Response Confidence: {confidence_level}

Answer the user's question:"""

    def _get_claims_template(self, response_mode: str) -> str:
        return """User's Question: {query}

CLAIMS INFORMATION IN THE POLICY:
{context}
{user_info}

INSTRUCTIONS:
1.  **Thought Process:**
    a. Identify all steps, deadlines, and required documents for filing a claim from the context.
    b. Organize these findings into a chronological sequence.
    c. Create a summary of all critical deadlines.
2.  **Answer Generation:**
    a. Present the claims process as a numbered list (Step 1, Step 2, ...).
    b. For each step, specify the action, any required documents, and the deadline.
    c. Add a separate "Key Deadlines" section for emphasis.
    d. Cite the source for each step and deadline.

METADATA: {metadata}
Response Confidence: {confidence_level}

Answer the user's question:"""

    def _get_eligibility_template(self, response_mode: str) -> str:
        return """User's Question: {query}

ELIGIBILITY REQUIREMENTS IN THE POLICY:
{context}
{user_info}

INSTRUCTIONS:
1.  **Thought Process:**
    a. Extract all eligibility criteria from the context (e.g., active work, minimum hours, waiting periods).
    b. Format these criteria as a checklist.
    c. If user information is available, compare it against the checklist items.
2.  **Answer Generation:**
    a. Present the requirements as a clear checklist.
    b. For each item, state the rule from the policy and cite the source.
    c. If user data is present, add a "Your Status" section and assess if the user seems to meet each criterion, explicitly stating that this is based only on the data provided.

METADATA: {metadata}
Response Confidence: {confidence_level}

Answer the user's question:"""

    def _get_admin_template(self, response_mode: str) -> str:
        return """User's Question: {query}

ADMINISTRATIVE INFORMATION IN THE POLICY:
{context}
{user_info}

INSTRUCTIONS:
1.  **Thought Process:**
    a. Identify the administrative process in the query (e.g., renewal, beneficiary change).
    b. Distinguish between tasks for the employee and tasks for the employer.
    c. Note any forms, contact points, or deadlines.
2.  **Answer Generation:**
    a. Clearly explain the process step-by-step.
    b. Use headings like "Your Responsibilities" and "Employer's Responsibilities" if applicable.
    c. Provide all relevant details like deadlines or required forms, citing sources for each.

METADATA: {metadata}
Response Confidence: {confidence_level}

Answer the user's question:"""

    def _get_general_template(self, response_mode: str) -> str:
        return """User's Question: {query}

RELEVANT POLICY INFORMATION:
{context}
{user_info}

INSTRUCTIONS:
1.  **Thought Process:**
    a. Read the user's question and break it down into its core components.
    b. Scan the context for any information relevant to these components.
    c. Organize the findings into logical themes.
2.  **Answer Generation:**
    a. Address each part of the user's question in a separate, well-organized paragraph.
    b. Ensure every statement is backed by information from the context and properly cited.
    c. If the context is vague, suggest a more specific follow-up question the user could ask.

METADATA: {metadata}
Response Confidence: {confidence_level}

Answer the user's question:"""

    def get_follow_up_suggestions(self, query_type: str, context_data: Dict[str, Any]) -> List[str]:
        """Generate follow-up question suggestions"""
        suggestions_map = {
            "benefits": [
                "How does my age affect the benefits?",
                "What additional benefits do I have for accidental death?",
                "When are the benefits reduced?"
            ],
            "definitions": [
                "What other important terms should I know?",
                "How does this definition apply in my case?",
                "Are there any exceptions to this definition?"
            ],
            "claims": [
                "What exact documents do I need for a claim?",
                "How long does it take to process a claim?",
                "Who can file a claim on my behalf?"
            ],
            "eligibility": [
                "When is the next enrollment period?",
                "What if I'm not eligible now?",
                "Can I add my dependents to the policy?"
            ],
            "policy_admin": [
                "How can I change my beneficiary?",
                "When can I make changes to my policy?",
                "How do I renew my coverage annually?"
            ]
        }
        return suggestions_map.get(query_type, [
            "Explain more about my life coverage.",
            "What other benefits does my policy include?",
            "What is the process for filing a claim?"
        ])