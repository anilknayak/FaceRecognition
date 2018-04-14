import os
import glob
import shutil
import json

basedir = os.path.abspath(os.path.dirname(__file__))
version = None
with open(os.path.join(basedir, "configuration/application/codefreeze.config"),'r') as config:
    configuration_data = config.readlines()
    configuration_details = ""
    for line in configuration_data:
        configuration_details += line
    configuration_details_json = json.loads(str(configuration_details))

    version = configuration_details_json['version']




def recursive_copy_files(source_path, destination_path, override=False):
    """
    Recursive copies files from source  to destination directory.
    :param source_path: source directory
    :param destination_path: destination directory
    :param override if True all files will be overridden otherwise skip if file exist
    :return: count of copied files
    """
    files_count = 0
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    items = glob.glob(source_path + '/*')
    for item in items:
        if os.path.isdir(item):
            path = os.path.join(destination_path, item.split('/')[-1])
            files_count += recursive_copy_files(source_path=item, destination_path=path, override=override)
        else:
            file = os.path.join(destination_path, item.split('/')[-1])
            if not os.path.exists(file) or override:
                shutil.copyfile(item, file)
                files_count += 1
    return files_count


def update_codefreeze():
    with open(os.path.join(basedir, "configuration/application/codefreeze.config"),'r+') as jsonFile:
        data = json.load(jsonFile)

        tmp = data["version"]
        data["version"] = int(tmp)+1

        jsonFile.seek(0)  # rewind
        json.dump(data, jsonFile)
        jsonFile.truncate()


todir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Release/V'+str(version)+"/"))

# if os.chdir(todir):
#     print("Present")
# else:
#     os.mkdir(todir)

recursive_copy_files(basedir, todir, True)

update_codefreeze()