module.exports = {
    devServer: {
        disableHostCheck: true,
        proxy: {
            '/api': {
                target: 'http://114.70.193.160:8787/api',
                changeOrigin: true,
                pathRewrite: {
                    '^/api': ''
                }
            }
        }
    },
    configureWebpack: {
        devServer: {
            host: '0.0.0.0',
            port: '8080'
        }
    }
}