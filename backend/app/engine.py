import dspy

class ScoutingReportSignature(dspy.Signature):
    """
    Expert Sporting Director signature for generating tactical scouting briefs.
    Analyzes player data and context to provide actionable recruitment intelligence.
    """
    context = dspy.InputField(desc="Markdown text containing layout-aware scouting data and metrics.")
    tactical_query = dspy.InputField(desc="Specific tactical question or requirement from the coaching staff.")
    scouting_brief = dspy.OutputField(desc="A comprehensive tactical scouting report with reasoning.")

class ScoutIntelRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought to force the model to reason before outputting the brief
        self.generate_report = dspy.ChainOfThought(ScoutingReportSignature)

    def forward(self, context_str: str, query_str: str):
        """
        Executes programmatic inference to generate a scouting report.
        """
        # The DSPy engine handles the prompt construction and CoT execution
        prediction = self.generate_report(context=context_str, tactical_query=query_str)
        return prediction
