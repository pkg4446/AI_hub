const fsys  = require('./fs_core');

function change(path,classnum){
    const directory = fsys.Dir(path);
    for (let index = 0; index < directory.length; index++) {
        const file_name = directory[index];
        const file = fsys.fileRead(path,file_name).split("\n");
        let change_class = ""
        for (let index1 = 0; index1 < file.length; index1++) {
            const roi = file[index1].split(" ");
            change_class += classnum;
            for (let index2 = 1; index2 < roi.length; index2++) {
                change_class += " "+roi[index2];
            }
            if(index1<file.length-1) change_class += "\n";
        }
        fsys.fileMK(path,change_class,file_name);
    }
}

change("./test",1);