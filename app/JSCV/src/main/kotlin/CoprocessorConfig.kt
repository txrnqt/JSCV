data class CoprocessorConfig(
    var hostname: String,
    var ip: String,
    var port: Int) {

    fun setServerConfig(hostname: String, ip: String, port: Int) {
        this.hostname = hostname
        this.ip = ip
        this.port = port
    }
}