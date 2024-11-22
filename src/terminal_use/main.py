from langchain_openai import ChatOpenAI

def init_model():
    return ChatOpenAI(model="gpt-4o-mini")

import sys
import subprocess
import json
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.styles import Style
STYLE = Style.from_dict({
    'header': '#00aa00 bold',
    'prompt': '#0000aa',
    'plan': '#888888',
    'next_action': '#00aa00',
})
from prompt_toolkit.formatted_text import FormattedText
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s -\n%(message)s')

SYSTEM_MESSAGE = (
    "You are a helpful assistant that can complete tasks in the terminal. "
    "You complete a task by multiple steps where each action describes a command to be issued. "
    "- Limitation 1: You cannot change directories, e.g. use the `cd` command. "
    "After each command, you will receive the output of the command. "
)

SYSTEM_MESSAGE_PLANNER = SYSTEM_MESSAGE + (
    "Now you are given the task to complete. "
    "The input is a dictionary with one key: 'task'. "
    "  - The 'task' key is the task to be performed. "
    "Based on the task description, you need to create a plan to accomplish the task. "
    "The plan should be a list of steps to be performed in natural language. "
    "The final action should be a way to verify that the task is complete. "
    "Your output should be a dictionary with one key: 'plan'. "
    "  - The 'plan' key is a list of steps to be performed in natural language. "
)

class PlannerOutputDict(BaseModel):
    plan: list[str] = Field(description="The plan to be performed")

    def __repr__(self):
        plan_str = ""
        for i, action in enumerate(self.plan):
            plan_str += f"    {i}. {action}"
            if i < len(self.plan) - 1:
                plan_str += "\n"
        return (
            f"PlannerOutputDict(\n"
            f"  plan: \n{plan_str}\n"
            f")"
        )

SYSTEM_MESSAGE_RUNNER = SYSTEM_MESSAGE + (
    "Now you are given the task to complete, the current plan for the task, the index of the current action, and a history of commands you have issued and their outputs. "
    "The input is a dictionary with four keys: 'task', 'plan', 'current_action', and 'history'. "
    "- The 'task' key is the task to be performed. "
    "- The 'plan' key is a list of natural language steps in the current plan. "
    "- The 'current_action' key is the index of the current action in the plan. "
    "- The 'history' key is a list of dictionaries where each dictionary has two keys: 'command' and 'output'. "
    "  - The 'command' key is a shell command that was issued. "
    "  - The 'output' key is the output of the command. "
    "Based on the task, the plan, the current action, and the history, determine the next command to be issued. "
    "Your output should be a dictionary with one key: 'command'. "
    "- The 'command' key is the next command to be issued. "
)

class RunnerOutputDcit(BaseModel):
    command: str = Field(description="The next command to be issued")

    def __repr__(self):
        return (
            f"RunnerOutputDcit(\n"
            f"  command: {self.command}\n"
            f")"
        )

SYSTEM_MESSAGE_LAST_STEP_INPUT = (
    "Now you are given the task to complete, the current plan for the task, the index of the last action, and a history of commands you have issued and their outputs. "
    "The input is a dictionary with four keys: 'task', 'plan', 'last_action', and 'history'. "
    "- The 'task' key is the task to be performed. "
    "- The 'plan' key is a list of natural language steps in the current plan. "
    "- The 'last_action' key is the index of the last action in the plan. "
    "- The 'history' key is a list of dictionaries where each dictionary has two keys: 'command' and 'output'. "
    "  - The 'command' key is a shell command that was issued. "
    "  - The 'output' key is the output of the command. "
)
SYSTEM_MESSAGE_VERIFIER = SYSTEM_MESSAGE + SYSTEM_MESSAGE_LAST_STEP_INPUT + (
    "Based on the task, the plan, the last action, and the history, verify if the task is complete. "
    "Your output should be a dictionary with one key: 'is_completed'. "
    "  - If completed, the 'is_completed' should be set to true. "
    "  - If not completed, the 'is_completed' should be set to false. "
)

class VerifierOutputDcit(BaseModel):
    is_completed: bool = Field(description="Whether the task is completed")

    def __repr__(self):
        return (
            f"VerifierOutputDcit(\n"
            f"  is_completed: {self.is_completed}\n"
            f")"
        )

SYSTEM_MESSAGE_REVISER = SYSTEM_MESSAGE + SYSTEM_MESSAGE_LAST_STEP_INPUT +(
    "Based on the task, the plan, the last action, and the history, revise the plan and point to the next action. "
    "Only revise the steps after the last action. "
    "Do not delete or change any steps before the last action, including the last action. "
    "The final action should be a way to verify that the task is complete. "
    "Your output should be a dictionary with two keys: 'revised_plan' and 'next_action'. "
    "  - The 'revised_plan' key is a list of natural language steps in the revised plan. "
    "  - The 'next_action' key is the index of the next action in the revised plan. "
)

class ReviserOutputDcit(BaseModel):
    revised_plan: list[str] = Field(description="The revised plan to be performed")
    next_action: int = Field(description="The index of the next action in the revised plan")
    def __repr__(self):
        plan_str = ""
        for i, action in enumerate(self.revised_plan):
            plan_str += f"    {i}. {action}"
            if i < len(self.revised_plan) - 1:
                plan_str += "\n"
        return (
            f"ReviserOutputDcit(\n"
            f"  revised_plan: \n{plan_str}\n"
            f"  next_action: {self.next_action}\n"
            f")"
        )

SYSTEM_MESSAGE_SIDE_EFFECT_CHECKER = SYSTEM_MESSAGE + (
    "Now you are given a command. "
    "Check if the command has side effects. "
    "- A command has side effects if it deletes, creates, or changes files. "
    "- A command does not have side effects if it only reads files. "
    "Your output should be a dictionary with one key: 'has_side_effects'. "
    "  - If the command has side effects, the 'has_side_effects' should be set to true. "
    "  - If the command does not have side effects, the 'has_side_effects' should be set to false. "
)

class SideEffectCheckerOutputDict(BaseModel):
    has_side_effects: bool = Field(description="Whether the command has side effects")

class DefaultConfig:
    skip_1st_check: bool = False
    execution_type: str = "smart"
    max_queries: int = 32

import click

@click.command()
@click.argument('task', type=str)
@click.option('-s', '--skip-1st-check', is_flag=True, default=False,
             help='skip the check in the first action')
@click.option('-e', '--execution-type', type=click.Choice(['smart', 'safe', 'all']), default=DefaultConfig.execution_type,
             help="execution type: smart mode (use LLM to tell if the command has side effects), safe mode (ask for all commands), or execute all commands (no asking). note this doesn't apply to the first action, which is controlled by -s,--skip-1st-check")
@click.option('-m', '--max-queries', type=int, default=DefaultConfig.max_queries,
             help='maximum number of queries to make')
def cli(task: str, skip_1st_check: bool, execution_type: str, max_queries: int):
    use_llm_to_perform_tasks_in_terminal(task, skip_1st_check, execution_type, max_queries)

def use_llm_to_perform_tasks_in_terminal(
    task: str,
    skip_1st_check: bool=DefaultConfig.skip_1st_check,
    execution_type: str=DefaultConfig.execution_type,
    max_queries: int=DefaultConfig.max_queries,
):
    logging.info(f"{task=}")

    model = init_model()
    logging.info(f"{model=}")

    # initial plan
    plan = model.with_structured_output(PlannerOutputDict).invoke([
        SystemMessage(content=SYSTEM_MESSAGE_PLANNER),
        HumanMessage(content=json.dumps({"task": task})),
    ]).plan
    
    current_step_idx = 0
    history = []

    for i in range(max_queries - 1):
        idict = {"task": task, "plan": plan, "current_action": current_step_idx, "history": history}
        logging.info(f"{idict=}")

        odict = model.with_structured_output(RunnerOutputDcit).invoke([
            SystemMessage(content=SYSTEM_MESSAGE_RUNNER),
            HumanMessage(content=json.dumps(idict)),
        ])
        logging.info(f"{odict=}")
        current_command = odict.command
        
        if len(history) > 0 and current_command == history[-1]["command"]:
            logging.info("command is the same as the previous command, skipping...")
            continue

        # create prompt session with custom style
        prompt_session = PromptSession(style=STYLE)

        # format plan with arrow pointing to next action
        plan_message = []
        for step_idx, action in enumerate(plan):
            prefix = "-> " if step_idx == current_step_idx else "   "
            plan_message.append(('class:plan' if step_idx != current_step_idx else 'class:next_action', 
                               f"{prefix}{action}\n"))

        # show command and get user input
        prompt_message = [
            ('class:header', '[Terminal Use]' + (" Task started!" if i == 0 else "") + '\n'),
            ('class:header', f"Task: {task}\n"),
            ('class:header', f"Plan:\n"),
            *plan_message,
            ('class:prompt', 'Edit command and press Enter to execute, `n` to skip, or Ctrl-C/Ctrl-D to exit:\n'),
            ('class:prompt', '> '),
        ]
        
        try:
            if i == 0 and not skip_1st_check:
                execute_without_prompt = False
            elif execution_type == "safe":
                execute_without_prompt = False
            elif execution_type == "smart":
                message = [
                    SystemMessage(content=SYSTEM_MESSAGE_SIDE_EFFECT_CHECKER),
                    HumanMessage(content=json.dumps({"command": current_command})),
                ]
                odict = model.with_structured_output(SideEffectCheckerOutputDict).invoke(message)
                execute_without_prompt = not odict.has_side_effects
            elif execution_type == "all":
                execute_without_prompt = True

            if execute_without_prompt:
                prompt_message = prompt_message[1:-2] if i > 0 else prompt_message[:-2]
                prompt_message.append(('class:prompt', f">> {current_command}"))
                print_formatted_text(FormattedText(prompt_message), style=STYLE)
                response = current_command
            else:
                response = prompt_session.prompt(prompt_message, default=current_command)

            if response.lower() != 'n':
                # execute the edited command and capture output
                try:
                    output = subprocess.check_output(response, shell=True, text=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    output = e.output

                logging.info(f"output:\n{output}")

                history.append({"command": response, "output": output})
            else:
                output = "[command skipped by user]"
        except KeyboardInterrupt:
            logging.info("exiting...")
            sys.exit(0)

        idict = {"task": task, "plan": plan, "last_action": current_step_idx, "history": history}
        logging.info(f"{idict=}")

        odict = model.with_structured_output(VerifierOutputDcit).invoke([
            SystemMessage(content=SYSTEM_MESSAGE_VERIFIER),
            HumanMessage(content=json.dumps(idict)),
        ])
        logging.info(f"{odict=}")
        
        if odict.is_completed:
            break
        
        odict = model.with_structured_output(ReviserOutputDcit).invoke([
            SystemMessage(content=SYSTEM_MESSAGE_REVISER),
            HumanMessage(content=json.dumps(idict)),
        ])
        plan = odict.revised_plan
        current_step_idx = odict.next_action

    print_formatted_text(FormattedText([('class:header', '[Terminal Use] Task completed!\n')]), style=STYLE)

if __name__ == "__main__":
    cli()