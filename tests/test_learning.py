"""本地离线验证：不发网络请求，不运行模型生成的代码。"""
import unittest
from types import SimpleNamespace
from hello_agents import SimpleAgent, ReActAgent, ReflectionAgent, ToolRegistry, Message
from hello_agents.agents.function_call_agent import FunctionCallAgent
from hello_agents.tools.builtin.calculator import CalculatorTool


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.provider = 'fake'
    def invoke(self, messages, **kwargs):
        self.requests.append(list(messages))
        return next(self.responses)


class LocalChecks(unittest.TestCase):
    def test_message_and_history(self):
        llm = FakeLLM(['你好'])
        agent = SimpleAgent(name='助手', llm=llm)
        self.assertEqual(agent.run('打招呼'), '你好')
        self.assertEqual([m.role for m in agent.get_history()], ['user','assistant'])
        self.assertEqual(Message('内容','user').to_dict(), {'role':'user','content':'内容'})
        agent.clear_history()
        self.assertEqual(agent.get_history(), [])

    def test_calculator_and_registry(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        self.assertEqual(registry.execute_tool('python_calculator','2+3*4'), '14')
        self.assertIn('错误', registry.execute_tool('not_registered','x'))

    def test_function_registry(self):
        registry = ToolRegistry()
        registry.register_function('double', '重复文本', lambda x: x*2)
        self.assertEqual(registry.execute_tool('double', 'ab'), 'abab')

    def test_simple_agent_tool_round(self):
        llm = FakeLLM(['[TOOL_CALL:python_calculator:2+3]', '结果是5'])
        registry = ToolRegistry(); registry.register_tool(CalculatorTool())
        agent = SimpleAgent(name='助手', llm=llm, tool_registry=registry)
        self.assertEqual(agent.run('计算'), '结果是5')
        self.assertIn('5', str(llm.requests[1]))

    def test_react_real_observation_in_next_prompt(self):
        llm = FakeLLM(['Thought: 计算\nAction: calc[2+3]', 'Thought: 完成\nAction: Finish[5]'])
        registry = ToolRegistry(); registry.register_function('calc','计算',lambda q:'5')
        agent = ReActAgent(name='助手', llm=llm, tool_registry=registry)
        self.assertEqual(agent.run('计算'), '5')
        self.assertIn('Observation: 5', llm.requests[1][0]['content'])

    def test_function_call_tool_result_matches_id(self):
        registry = ToolRegistry(); registry.register_function('calc','计算',lambda q:'5')
        call = SimpleNamespace(id='call_1', type='function', function=SimpleNamespace(name='calc',arguments='{"input":"2+3"}'))
        def response(content, calls):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content,tool_calls=calls))])
        responses=iter([response(None,[call]),response('5',[])])
        llm=FakeLLM([])
        agent=FunctionCallAgent(name='助手',llm=llm,tool_registry=registry)
        snapshots=[]
        def invoke(messages, **kwargs):
            snapshots.append(list(messages))
            return next(responses)
        agent._invoke_with_tools=invoke
        self.assertEqual(agent.run('计算'), '5')
        tools=[m for m in snapshots[1] if m['role']=='tool']
        self.assertEqual(tools[0]['tool_call_id'],'call_1')
        self.assertEqual(tools[0]['content'],'5')

    def test_reflection_improvement_path(self):
        llm=FakeLLM(['初稿','需要补充例子','改进稿'])
        agent=ReflectionAgent(name='助手',llm=llm,max_iterations=1)
        self.assertEqual(agent.run('写文章'),'改进稿')
        self.assertIn('需要补充例子',llm.requests[2][0]['content'])

    @unittest.expectedFailure
    def test_reflection_negation_should_not_stop(self):
        # 已知问题：子串匹配会把否定句误判为无需改进。
        llm=FakeLLM(['初稿','不能认为无需改进，建议补充例子','改进稿'])
        agent=ReflectionAgent(name='助手',llm=llm,max_iterations=1)
        self.assertEqual(agent.run('写文章'),'改进稿')


if __name__ == '__main__':
    unittest.main(verbosity=2)
