from langchain_openai import ChatOpenAI

def init_model():
    # initialize the model with gpt-4o-mini
    return ChatOpenAI(model="gpt-4o-mini")

import sys
import subprocess
import json
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.styles import Style
# define color styles for different parts of the output
STYLE = Style.from_dict({
    'header': '#00aa00 bold',
    'prompt': '#0000aa', 
    'plan': '#888888',
    'next_action': '#00aa00',
})
from prompt_toolkit.formatted_text import FormattedText
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s -\n%(message)s')

# base system message defining the assistant's role and limitations
SYSTEM_MESSAGE = (
    "You are a helpful assistant that can complete tasks in the terminal. "
    "You complete a task by multiple actions where each action either describes a shell command to be issued or a step that a LLM agent can perform. "
    "- Limitation 1: You cannot change directories, e.g. use the `cd` command. "
    "- Reminder 1: Always use absolute paths or relative paths to the current directory (due to Limitation 1). "
    "After each command, you will receive the output of the command. "
)

# system message for the planner component that creates the initial task plan
SYSTEM_MESSAGE_PLANNER = SYSTEM_MESSAGE + (
    "Now you are given the task to complete. "
    "The input is a dictionary with one key: 'task'. "
    "  - The 'task' key is the task to be performed. "
    "Based on the task description, you need to create a plan to accomplish the task. "
    "The plan should be a list of actions to be performed in natural language. "
    "The final action should be a way to verify that the task is complete. "
    "Your output should be a dictionary with one key: 'plan'. "
    "  - The 'plan' key is a list of actions to be performed in natural language. "
)

# pydantic model for planner output validation
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

# system message for the runner component that executes individual actions
SYSTEM_MESSAGE_RUNNER = SYSTEM_MESSAGE + (
    "Now you are given the task to complete, the current plan for the task, the index of the current action, and a history of commands you have issued and their outputs. "
    "The input is a dictionary with four keys: 'task', 'plan', 'current_action', and 'history'. "
    "- The 'task' key is the task to be performed. "
    "- The 'plan' key is a list of natural language actions in the current plan. "
    "- The 'current_action' key is the index of the current action in the plan. "
    "- The 'history' key is a list of dictionaries where each dictionary has two keys: 'command' and 'output'. "
    "  - The 'command' key is a shell command that was issued. "
    "  - The 'output' key is the output of the command. "
    "Based on the task, the plan, the current action, and the history, determine the next command to be issued. "
    "Do not repeat the last command in the history. "
    "Your output should be a dictionary with one key: 'command'. "
    "- The 'command' key is the next command to be issued. "
)

# pydantic model for runner output validation
class RunnerOutputDcit(BaseModel):
    command: str = Field(description="The next command to be issued")

    def __repr__(self):
        return (
            f"RunnerOutputDcit(\n"
            f"  command: {self.command}\n"
            f")"
        )

# shared input message for components that need task, plan, last action and history
SYSTEM_MESSAGE_LAST_STEP_INPUT = (
    "Now you are given the task to complete, the current plan for the task, the index of the last action, and a history of commands you have issued and their outputs. "
    "The input is a dictionary with four keys: 'task', 'plan', 'last_action', and 'history'. "
    "- The 'task' key is the task to be performed. "
    "- The 'plan' key is a list of natural language actions in the current plan. "
    "- The 'last_action' key is the index of the last action in the plan. "
    "- The 'history' key is a list of dictionaries where each dictionary has two keys: 'command' and 'output'. "
    "  - The 'command' key is a shell command that was issued. "
    "  - The 'output' key is the output of the command. "
)

# system message for the verifier component that checks task completion
SYSTEM_MESSAGE_VERIFIER = SYSTEM_MESSAGE + SYSTEM_MESSAGE_LAST_STEP_INPUT + (
    "Based on the task, the plan, the last action, and the history, verify if the task is complete. "
    "Only return success if the last verify action in the plan is completed. "
    "If there is no more action to be taken (i.e. the task cannot be successfully completed by issuing any more commands), the task is failed. "
    "Your output should be a dictionary with one key: 'task_status'. "
    "- If completed, the 'task_status' should be set to 1. "
    "- If not completed, the 'task_status' should be set to 0. "
    "- If failed, the 'task_status' should be set to -1. "
)

# pydantic model for verifier output validation
class VerifierOutputDcit(BaseModel):
    task_status: int = Field(description="The status of the task")

    def __repr__(self):
        return (
            f"VerifierOutputDcit(\n"
            f"  task_status: {self.task_status}\n"
            f")"
        )

# system message for the reviser component that updates the plan
SYSTEM_MESSAGE_REVISER = SYSTEM_MESSAGE + SYSTEM_MESSAGE_LAST_STEP_INPUT + (
    "Based on the task, the plan, the last action, and the history, first revise the plan. "
    "As a LLM agent, you are allowed to consume some actions in the plan as part of the revision. "
    "The outputs in the history are very helpful for revising the plan. "
    "Write down the revised plan before outputting it, read the plan, and identify the next action. "
    "Only revise the actions after the last action. "
    "Do not delete or change any actions before the last action, including the last action. "
    "The final action should be a way to verify that the task is complete. "
    "Your output should be a dictionary with two keys: 'revised_plan' and 'next_action'. "
    "- The 'revised_plan' key is a list of natural language actions in the revised plan. "
    "- The 'next_action' key is the index of the next action in the revised plan. "
)

# pydantic model for reviser output validation
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

# system message for checking if commands have side effects
SYSTEM_MESSAGE_SIDE_EFFECT_CHECKER = SYSTEM_MESSAGE + (
    "Now you are given a command. "
    "Check if the command has side effects. "
    "- A command has side effects if it deletes, creates, or changes files. "
    "- A command does not have side effects if it only reads files. "
    "Your output should be a dictionary with one key: 'has_side_effects'. "
    "- If the command has side effects, the 'has_side_effects' should be set to true. "
    "- If the command does not have side effects, the 'has_side_effects' should be set to false. "
)

# pydantic model for side effect checker output validation
class SideEffectCheckerOutputDict(BaseModel):
    has_side_effects: bool = Field(description="Whether the command has side effects")

import click
from dataclasses import dataclass

@dataclass
class Config:
    """configuration for terminal task execution"""
    skip_1st_check: bool = False
    execution_type: str = "smart"
    max_queries: int = 32

# CLI interface using click
@click.command()
@click.argument('task', type=str)
@click.option('-s', '--skip-1st-check', is_flag=True, default=Config.skip_1st_check,
             help='skip the check in the first action')
@click.option('-e', '--execution-type', type=click.Choice(['smart', 'safe', 'all']), default=Config.execution_type,
             help="execution type: smart mode (use LLM to tell if the command has side effects), safe mode (ask for all commands), or execute all commands (no asking). note this doesn't apply to the first action, which is controlled by -s,--skip-1st-check")
@click.option('-m', '--max-queries', type=int, default=Config.max_queries,
             help='maximum number of queries to make')
def cli(task: str, skip_1st_check: bool, execution_type: str, max_queries: int):
    use_llm_to_perform_tasks_in_terminal(task, skip_1st_check, execution_type, max_queries)

# main function that orchestrates the task execution
def use_llm_to_perform_tasks_in_terminal(
    task: str,
    skip_1st_check: bool=Config.skip_1st_check,
    execution_type: str=Config.execution_type, 
    max_queries: int=Config.max_queries,
):
    logging.info(f"{task=}")
    task_status = 0

    # initialize the language model
    model = init_model()
    logging.info(f"{model=}")

    # get initial plan from the planner
    plan = model.with_structured_output(PlannerOutputDict).invoke([
        SystemMessage(content=SYSTEM_MESSAGE_PLANNER),
        HumanMessage(content=json.dumps({"task": task})),
    ]).plan
    
    current_action_idx = 0
    history = []

    # main execution loop
    for i in range(max_queries - 1):
        idict = {"task": task, "plan": plan, "current_action": current_action_idx, "history": history}
        logging.info(f"{idict=}")

        # get next command from runner
        odict = model.with_structured_output(RunnerOutputDcit).invoke([
            SystemMessage(content=SYSTEM_MESSAGE_RUNNER),
            HumanMessage(content=json.dumps(idict)),
        ])
        logging.info(f"{odict=}")
        current_command = odict.command
        
        
        # skip if command is repeated
        if len(history) > 0 and current_command == history[-1]["command"]:
            logging.info("command is the same as the previous command, skipping...")
            continue


        # skip if command is repeated
        if len(history) > 0 and current_command == history[-1]["command"]:
            logging.info("command is the same as the previous command, skipping...")
            continue

        # setup interactive prompt session
        prompt_session = PromptSession(style=STYLE)

        # format plan display with arrow indicator
        plan_message = []
        for action_idx, action in enumerate(plan):
            prefix = "-> " if action_idx == current_action_idx else "   "
            plan_message.append(('class:plan' if action_idx != current_action_idx else 'class:next_action', 
                               f"{prefix}{action}\n"))

        # prepare display message
        prompt_message = [
            ('class:header', '[Terminal Use]' + (" Task started!" if i == 0 else "") + '\n'),
            ('class:header', f"Task: {task}\n"),
            ('class:header', f"Plan:\n"),
            *plan_message,
            ('class:prompt', 'Edit command and press Enter to execute, `n` to skip, or Ctrl-C/Ctrl-D to exit:\n'),
            ('class:prompt', '> '),
        ]
        
        try:
            # determine if command needs user confirmation
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

            # execute command with or without prompt
            if execute_without_prompt:
                prompt_message = prompt_message[1:-2] if i > 0 else prompt_message[:-2]
                prompt_message.append(('class:prompt', f">> {current_command}"))
                print_formatted_text(FormattedText(prompt_message), style=STYLE)
                response = current_command
            else:
                response = prompt_session.prompt(prompt_message, default=current_command)

            if response.lower() != 'n':
                # execute command and capture output
                try:
                    output = subprocess.check_output(response, shell=True, text=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    output = e.output

                print(output)

                history.append({"command": response, "output": output})
            else:
                output = "[command skipped by user]"
        except KeyboardInterrupt:
            logging.info("exiting...")
            sys.exit(0)

        # check if task is complete
        idict = {"task": task, "plan": plan, "last_action": current_action_idx, "history": history}
        logging.info(f"{idict=}")

        odict = model.with_structured_output(VerifierOutputDcit).invoke([
            SystemMessage(content=SYSTEM_MESSAGE_VERIFIER),
            HumanMessage(content=json.dumps(idict)),
        ])
        logging.info(f"{odict=}")
        task_status = odict.task_status
        
        if task_status != 0:
            break
        
        # revise plan if task not complete
        odict = model.with_structured_output(ReviserOutputDcit).invoke([
            SystemMessage(content=SYSTEM_MESSAGE_REVISER),
            HumanMessage(content=json.dumps(idict)),
        ])
        plan = odict.revised_plan
        current_action_idx = odict.next_action

    final_message = "Task completed!" if task_status == 1 else "Task failed!"
    print_formatted_text(FormattedText([('class:header', f'[Terminal Use] {final_message}\n')]), style=STYLE)

if __name__ == "__main__":
    cli()