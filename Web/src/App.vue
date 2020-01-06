<template>
  <div id="app">
    <el-header>
      <el-page-header title="손 랜드마크 검출 시스템">
      </el-page-header>
    </el-header>
    <el-main>
      <Output></Output>
    </el-main>
  </div>
</template>

<script>
  import Output from './components/Output'

export default {
  name: 'app',
  components: {
    Output
  }
}
</script>

<style>
  .el-page-header {
    border-bottom: 1px solid #e4e4e5;
    height: 60px;
  }
  .el-icon-back {
    visibility: hidden;
  }
  .el-page-header__title{
    margin-top: 15px;
    font-family: 나눔스퀘어라운드
  }
  .el-page-header__content{
    margin-top: 15px;
  }
</style>
