import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import gtm_agent.gtm_agent as agent_module


class FakeToolCallingModel(BaseChatModel):
    responses: list[AIMessage]
    response_index: int = 0
    _received_messages: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._received_messages.append(messages)
        response = self.responses[self.response_index]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class GtmAgentTests(unittest.TestCase):
    def test_agent_uses_application_tools_and_relevant_prompt(self):
        try:
            with patch("langchain.agents.create_agent") as create_agent:
                importlib.reload(agent_module)

            arguments = create_agent.call_args.kwargs
            self.assertEqual(
                [tool.name for tool in arguments["tools"]],
                [
                    "lookup_offering",
                    "build_prospect_profile",
                    "get_prospect",
                    "send_prospect_email",
                    "score_prospect",
                    "update_prospect_info",
                    "get_current_rep",
                ],
            )
            self.assertEqual(arguments["system_prompt"], agent_module.SYSTEM_PROMPT)
            self.assertIn("Ask a concise follow-up only when needed", agent_module.SYSTEM_PROMPT)
            self.assertIn("until the rep's request is complete", agent_module.SYSTEM_PROMPT)
        finally:
            importlib.reload(agent_module)

    def test_run_agent_preserves_invoke_configuration(self):
        fake_agent = Mock()
        fake_agent.invoke.return_value = {"messages": [AIMessage(content="Done.")]}

        with patch.object(agent_module, "gtm_agent", fake_agent):
            result = agent_module.run_agent(
                "Email this prospect", user_id="REP-1001", environment="test", thread_id="thread-1"
            )

        invocation, = fake_agent.invoke.call_args.args
        configuration = fake_agent.invoke.call_args.kwargs["config"]
        self.assertEqual(invocation, {"messages": [{"role": "user", "content": "Email this prospect"}]})
        self.assertEqual(configuration["run_name"], "GTM Assistant")
        self.assertEqual(str(configuration["run_id"]), result["run_id"])
        self.assertEqual(
            configuration["metadata"],
            {
                "thread_id": "thread-1",
                "user_id": "REP-1001",
                "environment": "test",
                "request_intent": "send_email",
            },
        )
        self.assertEqual(result["reply"], "Done.")

    def test_runtime_tools_execute_through_agent_tool_node(self):
        rep = agent_module.REP_IDS[0]
        tool_calls = [
            {"name": "get_current_rep", "args": {}, "id": "rep-call", "type": "tool_call"},
            {
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"name": "Taylor Example", "email": "taylor@example.com"},
                    "subject": "Following up",
                    "body": "Thanks for your time.",
                },
                "id": "email-call",
                "type": "tool_call",
            },
        ]
        fake_model = FakeToolCallingModel(
            responses=[
                AIMessage(content="", tool_calls=tool_calls),
                AIMessage(content="Email sent."),
            ]
        )
        test_agent = create_agent(
            model=fake_model,
            tools=[
                agent_module.lookup_offering,
                agent_module.build_prospect_profile,
                agent_module.get_prospect,
                agent_module.send_prospect_email,
                agent_module.score_prospect,
                agent_module.update_prospect_info,
                agent_module.get_current_rep,
            ],
            system_prompt=agent_module.SYSTEM_PROMPT,
        )

        result = test_agent.invoke(
            {"messages": [{"role": "user", "content": "Send the email."}]},
            config={"metadata": {"user_id": rep["rep_id"]}},
        )
        outputs = {
            message.tool_call_id: json.loads(message.content)
            for message in result["messages"]
            if hasattr(message, "tool_call_id")
        }

        self.assertEqual(outputs["rep-call"]["rep"], rep)
        self.assertEqual(outputs["email-call"]["status"], "sent")
        self.assertEqual(outputs["email-call"]["from"], rep["email"])
        self.assertEqual(
            set(test_agent.nodes["tools"].bound.tools_by_name),
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )
        self.assertEqual(fake_model._received_messages[0][0].content, agent_module.SYSTEM_PROMPT)
        self.assertNotIn("write_todos", fake_model._received_messages[0][0].content)


if __name__ == "__main__":
    unittest.main()
