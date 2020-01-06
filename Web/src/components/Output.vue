<template>
    <div class="main">
        <el-steps :active="active">
            <el-step title="Step 1" description="버튼을 눌러 사진을 업로드하세요."></el-step>
            <el-step title="Step 2" description="사진을 저장하세요."></el-step>
        </el-steps>
        <el-row class="output-main">
            <el-col :span="12" class="image-block">
                <div class="demo-image__placeholder">
                    <div class="block">
                        <el-image class="viewer" :src="src" alt="" v-loading="loading">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-picture-outline"></i>
                            </div>
                        </el-image>
                    </div>
                </div>
            </el-col>
            <el-col :span="11" class="image-upload">
                <el-row>
                    <el-card shadow="always" class="demonstration">
                        <h1>손 랜드마크 검출</h1><p> 소아의 내분비 문제와 성장 장애의 진단 및 치료를 위해 손 엑스레이 영상을 이용한 뼈 나이 측정이 필요합니다. 뼈 나이 측정을 위해 TW 방법을 사용할 경우, 손 엑스레이 영상에서 특정 랜드마크의 위치를 알아내야 합니다.
                            입력으로 손 엑스레이 영상이 들어오면 랜드마크의 위치를 찾아내어 출력합니다.</p>
                    </el-card>
                </el-row>
                <el-row>                
                    <el-upload
                            class="upload-demo"
                            drag
                            action=""
                            ref="upload"
                            :show-file-list="false"
                            :http-request="upload">
                        <i class="el-icon-upload"></i>
                        <div class="el-upload__text"><em>눌러서 사진을 업로드하세요.</em></div>
                    </el-upload>
                </el-row>
                <el-button type="primary" class="save" :disabled="this.button" @click="savePicture">사진 저장</el-button>
            </el-col>

            <span></span>
        </el-row>
    </div>
</template>

<script>
    const assets = require('../assets/5599.jpg');

    export default {
        name: "Output",
        data() {
            return {
                src: '',
                image: '',
                button: 'true',
                loading: false,
                active: 1,
                assets
            }
        },
        methods: {
            upload() {
                const formData = new FormData();
                const file = this.$refs.upload.uploadFiles[0];
                const headerConfig = { headers: { 'Content-Type': 'multipart/form-data' , 'responseType': 'blob'}};
                formData.append('file', file.raw);
                this.loading = true;
                this.$axios.post('/api/fileUpload', formData, headerConfig).then(res => {
                    this.src = "data:image/png;base64," + res.data;
                    this.$refs.upload.clearFiles();
                    this.button = false;
                    this.loading= false;
                    this.active = 2;
                })
                .catch(() => {
                    this.loading = false;
                    this.$refs.upload.clearFiles();
                })
            },
            savePicture() {
                let imgData = atob(this.src.split(",")[1]),
                    len = imgData.length,
                    buf = new ArrayBuffer(len),
                    view = new Uint8Array(buf),
                    blob,
                    i;
                const fileName = "result";

                for (i = 0; i < len; i++) {
                    view[i] = imgData.charCodeAt(i) & 0xff // masking
                }

                blob = new Blob([view], {
                    type: "application/octet-stream"
                });

                if (window.navigator.msSaveOrOpenBlob) {
                    window.navigator.msSaveOrOpenBlob(blob, fileName)
                } else {
                    //var url = URL.createObjectURL(blob);
                    var a = document.createElement("a")
                    a.style = "display: none"
                    //a.href = url;
                    a.href = this.src
                    a.download = fileName
                    document.body.appendChild(a)
                    a.click()

                    setTimeout(function() {
                        document.body.removeChild(a)
                        //URL.revokeObjectURL(url);
                    }, 100)
                }
            }
        }
    }
</script>

<style scoped>
    .main {
        width: 315mm;
        height: 435mm;
        margin: 0 auto 20px;
        position: relative;
        float: top;
    }
    .output-main {
        padding-left: 100px;
    }
    .demonstration {
        border: #e4e4e5 1px solid;
        padding: 10px;
    }
    .image-block {
        margin: 5px;
    }
    .image-upload {
        margin-top: 5px;
        margin-left: 5px;
        width: 365px;
        font-family: "나눔스퀘어라운드"
    }
    .output {
        border-bottom: #e4e4e5 1px solid;
        margin-bottom: 5px;
        margin-top: 5px;
        font-family: "나눔스퀘어라운드 Bold"
    }
    .demonstration {
        margin: 3px;
    }
    .viewer {
        margin: 5px;
    }
    .image-block {
        border: #e4e4e5 1px solid;
        border-radius: 10px;
    }
    .el-image {
    }
    .el-icon-picture-outline {
        width: 533.266px;
        height: 699.906px;
    }
    .save{
        width: 365px;
        margin-top: 10px;
    }
    .el-upload-list {
        visibility: hidden;
    }
</style>