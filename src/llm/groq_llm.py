from typing import List, Dict, Optional
import groq
import json
from dataclasses import dataclass

@dataclass
class CodeContext:
    """Represents code context for LLM."""
    content: str
    language: str
    file_path: str
    related_files: List[str]
    
class GroqLLM:
    """Groq-based LLM for code understanding and explanation."""
    
    def __init__(self, api_key: str):
        self.client = groq.Client(api_key=api_key)
        self.model = "mixtral-8x7b-32768"  # Using Mixtral for better code understanding
        
        # System prompt optimized for code understanding
        self.system_prompt = """You are an expert software development assistant with deep knowledge of multiple programming languages and software architectures. Your role is to help users understand and work with code across different languages and frameworks.

Key Capabilities:
1. Code Analysis:
   - Parse and understand code in multiple languages (Python, JavaScript, Java, C++, Go, etc.)
   - Identify design patterns, architectural choices, and best practices
   - Recognize language-specific idioms and conventions

2. Context Awareness:
   - Consider file relationships and dependencies
   - Understand project structure and organization
   - Track code flow across multiple files

3. Explanation Style:
   - Provide clear, concise explanations
   - Use relevant code examples when helpful
   - Break down complex concepts into digestible parts
   - Reference official documentation and best practices

4. Problem Solving:
   - Suggest improvements and optimizations
   - Identify potential issues and edge cases
   - Provide actionable solutions with examples

When analyzing code:
1. First understand the context and relationships between files
2. Consider the programming language's specific features and conventions
3. Look for patterns and architectural decisions
4. Explain both what the code does and why it's designed that way

Remember:
- Be precise and technical but accessible
- Consider security implications
- Highlight best practices and potential improvements
- Maintain awareness of the full codebase context

Format your responses in clear sections:
1. Summary (brief overview)
2. Detailed Analysis (main explanation)
3. Key Points (important takeaways)
4. Suggestions (if applicable)"""

    async def analyze_code(
        self,
        query: str,
        code_contexts: List[CodeContext],
        temperature: float = 0.1
    ) -> str:
        """
        Analyze code based on query and provided contexts.
        
        Args:
            query: User's question about the code
            code_contexts: List of relevant code snippets with metadata
            temperature: Control randomness (lower = more focused)
        """
        # Build context message
        context_msg = "Here are the relevant code sections:\n\n"
        for ctx in code_contexts:
            context_msg += f"File: {ctx.file_path} (Language: {ctx.language})\n"
            context_msg += f"Related files: {', '.join(ctx.related_files)}\n"
            context_msg += "```" + ctx.language + "\n"
            context_msg += ctx.content + "\n```\n\n"
            
        # Create chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context_msg + "\nQuestion: " + query}
        ]
        
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )
        
        return completion.choices[0].message.content
        
    async def suggest_improvements(
        self,
        code_context: CodeContext,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Suggest code improvements for given context.
        
        Args:
            code_context: Code to analyze
            focus_areas: Optional specific areas to focus on (e.g., ["performance", "security"])
        """
        focus_msg = ""
        if focus_areas:
            focus_msg = f"\nPlease focus on these areas: {', '.join(focus_areas)}"
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Please analyze this code and and aner questions for user understanding and clear any issues:{focus_msg}

File: {code_context.file_path}
Language: {code_context.language}

```{code_context.language}
{code_context.content}
```"""}
        ]
        
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=2048
        )
        
        return completion.choices[0].message.content
        
    async def explain_architecture(
        self,
        code_contexts: List[CodeContext],
        focus: Optional[str] = None
    ) -> str:
        """
        Explain the architecture and relationships between code components.
        
        Args:
            code_contexts: List of related code snippets
            focus: Optional specific aspect to focus on
        """
        context_msg = "Please explain the architecture of these related components:\n\n"
        for ctx in code_contexts:
            context_msg += f"File: {ctx.file_path} (Language: {ctx.language})\n"
            context_msg += "```" + ctx.language + "\n"
            context_msg += ctx.content + "\n```\n\n"
            
        if focus:
            context_msg += f"\nPlease focus specifically on: {focus}"
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context_msg}
        ]
        
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=3072
        )
        
        return completion.choices[0].message.content
