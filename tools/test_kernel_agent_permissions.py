"""
Unit tests for the interactive permission-approval loop in KernelAgent.

Covers the headless-stall fix: when opencode asks for permission
(``action.action=ask``), the harness services the request via the keyboard
instead of hanging forever. The opencode REST client is mocked — no server,
no GPU required:

    python -m unittest tools.test_kernel_agent_permissions -v
"""

import sys
import unittest.mock
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from tools.kernel_agent import KernelAgent  # noqa: E402


def _make_request(rid="per_1", session_id="ses_x", permission="bash",
                  patterns=("python *",), command="python bench.py"):
    """A permission-request payload as returned by ``GET /permission``."""
    return {
        "id": rid,
        "sessionID": session_id,
        "permission": permission,
        "patterns": list(patterns),
        "metadata": {"command": command},
        "always": [],
    }


def _make_agent(pending, answers):
    """KernelAgent wired to a mocked client and a scripted keyboard."""
    with unittest.mock.patch("tools.kernel_agent.OpenCodeClient") as cls:
        agent = KernelAgent(interactive_permissions=True)
    agent.client = unittest.mock.Mock()
    agent.client.list_permissions.return_value = list(pending)
    agent._session_id = "ses_x"
    answer_iter = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(answer_iter)
        except StopIteration as exc:  # safety: unexpected extra keyboard read
            raise AssertionError(f"keyboard read exhausted answers: {prompt}") from exc

    input_patch = unittest.mock.patch("builtins.input", side_effect=fake_input)
    input_patch.start()
    return agent, input_patch


class KeyboardApproveTest(unittest.TestCase):
    """The raw key reader maps input to opencode reply verbs."""

    def _map(self, key):
        with unittest.mock.patch("builtins.input", return_value=key):
            return KernelAgent._keyboard_approve(_make_request())

    def test_default_enter_is_once(self):
        self.assertEqual(self._map(""), "once")

    def test_y_is_once(self):
        self.assertEqual(self._map("y"), "once")

    def test_a_is_always(self):
        self.assertEqual(self._map("a"), "always")

    def test_n_is_reject(self):
        self.assertEqual(self._map("n"), "reject")

    def test_eof_is_reject(self):
        with unittest.mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(KernelAgent._keyboard_approve(_make_request()), "reject")

    def test_invalid_then_valid(self):
        with unittest.mock.patch("builtins.input", side_effect=["wat", "y"]):
            self.assertEqual(KernelAgent._keyboard_approve(_make_request()), "once")


class ServicePermissionAsksTest(unittest.TestCase):
    """The poll-time service answers each pending request exactly once."""

    def test_single_request_approved(self):
        agent, patch = _make_agent([_make_request()], ["y"])
        self.addCleanup(patch.stop)
        agent._service_permission_asks()
        agent.client.reply_permission.assert_called_once_with("per_1", "once")

    def test_always_mapping(self):
        agent, patch = _make_agent([_make_request()], ["a"])
        self.addCleanup(patch.stop)
        agent._service_permission_asks()
        agent.client.reply_permission.assert_called_once_with("per_1", "always")

    def test_reject_mapping(self):
        agent, patch = _make_agent([_make_request()], ["n"])
        self.addCleanup(patch.stop)
        agent._service_permission_asks()
        agent.client.reply_permission.assert_called_once_with("per_1", "reject")

    def test_request_for_other_session_ignored(self):
        agent, patch = _make_agent(
            [_make_request(session_id="ses_other")], []
        )
        self.addCleanup(patch.stop)
        agent._service_permission_asks()
        agent.client.reply_permission.assert_not_called()

    def test_same_id_asked_only_once(self):
        agent, patch = _make_agent([_make_request()], ["y"])
        self.addCleanup(patch.stop)
        agent._service_permission_asks()
        agent._service_permission_asks()  # poll again, request briefly still listed
        self.assertEqual(agent.client.reply_permission.call_count, 1)

    def test_list_error_is_silently_ignored(self):
        agent, patch = _make_agent([], [])
        self.addCleanup(patch.stop)
        agent.client.list_permissions.side_effect = RuntimeError("boom")
        agent._service_permission_asks()  # must not raise
        agent.client.reply_permission.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
