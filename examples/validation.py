"""Example: Semantic response validation."""
import asyncio
from promptguard import PromptChain, validators


async def main():
    """Execute with semantic validation."""
    
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        validators=[
            validators.length_range(
                min_chars=100,
                max_chars=500
            ),
            validators.contains_keywords(
                keywords=["risk", "mitigation", "strategy"],
                require_all=True
            ),
            validators.has_citations(required=True),
            validators.sentiment_check(
                allowed=["neutral", "professional"]
            )
        ],
        validation_mode="strict"  # Retry if any validator fails
    )
    
    prompt = """Provide a risk assessment for adopting cloud infrastructure. 
    Include:
    - Key risks
    - Mitigation strategies
    - References"""
    
    result = await chain.execute(prompt)
    
    print(f"Response: {result.response}")
    print(f"\nValidation Results:")
    for validator_name, validation in result.validation_results.items():
        status = "Success" if validation['passed'] else "Failed"
        print(f"  {status} {validator_name}: {validation['message']}")


if __name__ == "__main__":
    asyncio.run(main())
