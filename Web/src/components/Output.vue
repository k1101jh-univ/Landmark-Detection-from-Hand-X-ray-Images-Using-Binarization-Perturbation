<template>
    <el-row>
        <el-col :span="12" class="image-block">
            <div class="demo-image__placeholder">
                <div class="block">
                    <div class="output">
                        <span class="demonstration">output</span>
                    </div>
                    <img class="viewer" :src="src" alt="">
                </div>
            </div>
        </el-col>
        <el-col :span="11" class="image-upload">
            <el-upload
                    class="upload-demo"
                    drag
                    action=""
                    ref="upload"
                    :http-request="upload"
                    :on-preview="handlePreview">
                <i class="el-icon-upload"></i>
                <div class="el-upload__text">Drop file here or <em>click to upload</em></div>
            </el-upload>
        </el-col>

        <span></span>
    </el-row>
</template>

<script>
    const assets = require('../assets/5599.jpg');

    export default {
        name: "Output",
        data() {
            return {
                src: '',
                image: '',
                assets
            }
        },
        methods: {
            upload() {
                const formData = new FormData();
                const file = this.$refs.upload.uploadFiles[0];
                const headerConfig = { headers: { 'Content-Type': 'multipart/form-data' , 'responseType': 'blob'}};
                formData.append('file', file.raw);
                this.$axios.post('/api/fileUpload', formData, headerConfig).then(res => {
                    this.src = "data:image/png;base64," + res.data;
                    this.$refs.upload.clearFiles();
                })
                .catch(() => {
                    this.$refs.upload.clearFiles();
                })
            }
        }
    }
</script>

<style scoped>
    .image-block {
        margin: 5px;
    }
    .image-upload {
        margin-top: 5px;
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
</style>