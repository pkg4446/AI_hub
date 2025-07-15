const fsys  = require('./fs_core');
const fs    = require('fs');

function path_tree(path){
    const list = fsys.Dir(path);
    const response = {};
    for (let index = 0; index < list.length; index++) {
        const directory = fsys.isDirectory(path+"/"+list[index]);
        if(directory){
            const result = path_tree(path+"/"+list[index],fsys.Dir(path+"/"+list[index]));
            for (const directory in result) {
                response[directory]=result[directory];
            }
        }else{
            if(response[path]==undefined) response[path]=[];
            response[path].push(list[index]);
        }
    }
    return response;    
}

module.exports = {    
    convert: function(text){
        try {
            const path_annotation = "./annotation";
            const response = {};
            
            const tree = path_tree(path_annotation);
            console.log(tree);
            for (const directory in tree) {
                const files = tree[directory];
                for (let index = 0; index < files.length; index++) {
                    const file = files[index];
                    const data = fsys.fileRead(directory,file);
                    if(data){
                        try {
                            const jsonData = JSON.parse(data);
                            const imageWidth = jsonData.IMAGE.WIDTH;
                            const imageHeight = jsonData.IMAGE.HEIGHT;
                            const imageFileName = jsonData.IMAGE.IMAGE_FILE_NAME;

                            // '알'에 대한 클래스 ID는 0으로 가정합니다.
                            const classId = 0; 

                            let yoloAnnotations = [];

                            jsonData.ANNOTATION_INFO.forEach(annotation => {
                                const xtl = annotation.XTL;
                                const ytl = annotation.YTL;
                                const xbr = annotation.XBR;
                                const ybr = annotation.YBR;

                                // 바운딩 박스의 너비와 높이 계산
                                const width = xbr - xtl;
                                const height = ybr - ytl;

                                // 바운딩 박스의 중심 좌표 계산
                                const x_center = xtl + (width / 2);
                                const y_center = ytl + (height / 2);

                                // 이미지 크기에 맞춰 정규화
                                const x_center_normalized = x_center / imageWidth;
                                const y_center_normalized = y_center / imageHeight;
                                const width_normalized = width / imageWidth;
                                const height_normalized = height / imageHeight;

                                // YOLO 형식 문자열 생성 (소수점 6자리까지 표현)
                                yoloAnnotations.push(
                                    `${classId} ${x_center_normalized.toFixed(6)} ${y_center_normalized.toFixed(6)} ${width_normalized.toFixed(6)} ${height_normalized.toFixed(6)}`
                                );
                            });

                            const outputDir = directory.replace(path_annotation,"./yolo_labels");
                            if(!fsys.check(outputDir)){
                                fsys.folderMK(outputDir);
                            }

                            // // 결과 `.txt` 파일 경로 설정
                            const outputTxtFileName = imageFileName.replace(/\.[^/.]+$/, "") + '.txt'; // 확장자 제거하고 .txt 붙이기
                            const outputTxtFilePath = outputDir+"/"+outputTxtFileName;

                            // 파일 쓰기
                            fs.writeFile(outputTxtFilePath, yoloAnnotations.join('\n'), 'utf8', (err) => {
                                if (err) {
                                    console.error('YOLO 형식 파일을 쓰는 중 오류가 발생했습니다:', err);
                                    return;
                                }
                                console.log(`YOLO 형식 데이터가 ${outputTxtFilePath} 파일에 성공적으로 저장되었습니다.`);
                            });
                            
                        } catch (error) {
                            console.error('JSON 데이터를 파싱하는 중 오류가 발생했습니다:', parseError);
                        }
                    } else {
                        console.error(`Failed to read file: ${directory}/${file}`);
                    }
                    
                }
            }
            console.log(text,response);
            return response;
        } catch (error) {
            console.error(error);
            return false;
        }
    },
}