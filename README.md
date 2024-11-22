# Terminal Use

Ask a task to complete in the terminal, e.g. "delete the PDF files unrelated to AI in the current directory".
Terminal Use will
- query an LLM to generate an action
- run the command needed by the action and get the output
- iterate until the task is complete

## Usage

```bash
terminal-use "delete the pdf files unrelated to ai in the current directory"
```

```bash
terminal-use "commit the modified python files via git"
```
