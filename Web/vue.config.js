module.exports = {
    devServer: {
        proxy: {
            '/api': {
                target: 'http://114.70.193.160:8787/api',
                changeOrigin: true,
                pathRewrite: {
                    '^/api': ''
                }
            }
        }
    }
}