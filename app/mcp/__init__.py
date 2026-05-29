"""Model Context Protocol (MCP) layer.

Two consumers share one tool definition (``app.mcp.tools``):

* ``app.mcp.server``     — exposes the tools over the MCP protocol (FastMCP) so
  external agents (Claude Desktop, a teammate's service) can call them.
* ``app.agent.runtime``  — the production agent builds a ``ToolRegistry`` and
  dispatches the same tools in-process, so the bot never makes a network hop to
  call its own tools. (The older ``app.chatbot.agent`` MCPAgent used the same
  registry but is retired along with ``ChatGraphRunner``.)
"""
