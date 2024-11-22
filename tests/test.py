import os
import shutil
from terminal_use.main import use_llm_to_perform_tasks_in_terminal

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "tmp")
    
    # delete and recreate tmp folder
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # test 1
    file_to_init = os.path.join(test_dir, "machine-learning-book.txt")
    with open(file_to_init, "w") as f:
        pass
    
    use_llm_to_perform_tasks_in_terminal(
        '''there is a file under "{}" that is related to AI, '''
        '''rename it to "ai-materials" (without changing the extension)'''.format(test_dir),
        skip_1st_check=True,
        execution_type="all",
    )

    # test 2
    file_to_init = os.path.join(test_dir, "statistics-materials.txt")
    with open(file_to_init, "w") as f:
        pass

    file_to_init = os.path.join(test_dir, "machine-learning-materials.txt")
    with open(file_to_init, "w") as f:
        pass

    use_llm_to_perform_tasks_in_terminal(
        '''clean up the files under "{}" by removing common suffixes from the filenames. '''
        '''don't remove the extension!!!'''.format(test_dir),
        skip_1st_check=True,
        execution_type="all",
    )