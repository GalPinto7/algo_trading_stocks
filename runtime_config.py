"""
Because we edit on locally but run the code on Google colab, because of the lack on local GPU
we have modified the paths of the files based on where we work.
"""

def get_where_the_code_runs() -> int:
    """
    We need to receive where we run the code, locally or in google colab.
    That allows us the change the paths of the files and so on accordingly.
    1 -> locally.
    2 -> google colab.
    :return:  The number of where we run the code.
    """
    where_the_code_runs = int(input('please enter where the code runs. \nenter 1 for locally, 2 for Google colab: '))

    if(where_the_code_runs not in [1,2]):
        raise ValueError("The number must be 1 or 2")

    return where_the_code_runs

# Default value for imports.
# This prevents input() from running when another file imports runtime_config.
where_the_code_runs = 1

if __name__ == "__main__":
    where_the_code_runs = get_where_the_code_runs()
    print(where_the_code_runs)
