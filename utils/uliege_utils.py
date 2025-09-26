import os

def get_name(path):
    end_c = path.rfind("/")
    return path[end_c+1:]

# Get the class of an image from its path
# ! Only works when the path is in the form: /home/.../project/class/image_name
def get_class(path):
    end_c = path.rfind("/")
    begin_c = path.rfind("/", 0, end_c) + 1
    if end_c == -1 or begin_c == -1:
        return -1
    return path[begin_c:end_c]

# Get the project name from the path
def get_proj(path):
    end_c = path.rfind("/")
    begin_c = path.rfind("/", 0, end_c) + 1
    end_proj = path[begin_c:end_c].rfind("_")
    proj_name = path[begin_c:begin_c+end_proj]
    if proj_name.rfind("_") != -1:
        rest = proj_name[proj_name.rfind("_")+1: len(proj_name)]
        if rest.isdigit():
                proj_name = proj_name[0:proj_name.rfind("_")]
    return proj_name

# Change class names to have all class number starting from 0
def rename_classes(class_list):
    classes = os.listdir(class_list)
    classes.sort()
    cpt_c = 0
    old_name = ""
    new_classes = []
    for c in classes:
        idx = c.rfind("_")
        c_project = c[0:idx]
        if c_project.rfind("_") != -1:
            rest = c_project[c_project.rfind("_")+1: len(c_project)]
            if rest.isdigit():
                c_project = c_project[0:c_project.rfind("_")]
        if c_project == old_name:
            cpt_c += 1
        else: 
            cpt_c = 0

        new_classes.append(c_project + "_" + str(cpt_c))
        old_name = c_project
    return new_classes

# Get new class name of a class
def get_new_name(class_name, path=None):
    classes = os.listdir(path)
    classes.sort()
    proj = get_proj(class_name)
    cpt_c = 0
    for c in classes:
        if c == class_name:
            return proj + "_" + str(cpt_c)
        elif proj in c:
            cpt_c += 1
    return -1 

