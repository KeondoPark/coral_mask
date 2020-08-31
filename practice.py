import pathlib
for index, file in enumerate(pathlib.Path('for_evaluation(test_set)/image').iterdir()):
    print(file.resolve())
    print(index)