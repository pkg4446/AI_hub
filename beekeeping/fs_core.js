const fs    = require('fs');

module.exports = {
    Dir:    function(FOLDER){
        try {
            const PATH  = FOLDER + "/";
            const dir   = fs.readdirSync(PATH);
            return dir;
        } catch (error) {    
            return false;
        }
    },

    isDirectory: function(path) {
        try {
          const stats = fs.statSync(path);
          return stats.isDirectory();
        } catch (error) {
          return false; // 에러 발생 시 디렉토리로 간주하지 않음 (경로가 존재하지 않거나 접근 권한 없을 때)
        }
    },

    check:  function(FOLDER){
        const PATH  = FOLDER;
        const CHECK = fs.existsSync(PATH, 'utf8')
        return CHECK;
    },
    
    move:   async function(TARGET,MOVE){
        try {
            fs.renameSync(TARGET, MOVE);
            return true;
        } catch (error) {    
            return false;
        }
    },

    folderDel:   function(FOLDER){
        const PATH  = FOLDER + "/";
        try {
            fs.rmSync(PATH, { recursive: true, force: true });
        } catch (error) {   
            console.error(`${FOLDER} 폴더가 삭제되었습니다.`);
        }
    },

    folderMK:    function(PATH){  
        const path_forder = PATH.split("/")
        let   response = false;
        if(path_forder[0]=="."){
            response = true;
            let path_make = ".";
            for (let index = 1; index < path_forder.length; index++) {
                path_make += "/"+path_forder[index];
                console.log(path_make);
                if(!fs.existsSync(path_make, 'utf8'))fs.mkdirSync(path_make);
            }
        }
        return response;
    },

    fileRead:    function(FOLDER,FILE){ 
        let response;  
        try {
            response = fs.readFileSync(`${FOLDER}/${FILE}`, 'utf8'); 
        } catch (error) {
            response = null;
        }
        return response;
    },
    
    fileMK:    function(FOLDER,DATA,FILE){        
        try {
            fs.writeFileSync(`${FOLDER}/${FILE}`, DATA);
        } catch (error) {
            return false;
        }
        return true;
    },

    fileADD:    function(FOLDER,DATA,FILE){     
        try {
            fs.appendFileSync(`${FOLDER}/${FILE}`, DATA);
        } catch (error) {
            return false;
        }
        return true;
    },

    fileDel:    function(FOLDER,FILE){
        const PATH = FOLDER + "/";
        try {
            fs.readdirSync(PATH, 'utf8');
            try {
                fs.unlinkSync(PATH+FILE);
                return FILE;
            } catch (error) {
                return false;
            }            
        } catch (error) {   
            fs.mkdirSync(PATH);
            return false;
        }                
    },
}