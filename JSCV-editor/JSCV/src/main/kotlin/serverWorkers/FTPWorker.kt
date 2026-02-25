package serverWorkers

import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPReply
import javax.swing.SwingWorker
import java.io.File

class FTPWorker {
    private var ftp: FTPClient? = null
    private var isConnected = false

    fun connect(
        host: String,
        username: String,
        password: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        object : SwingWorker<Unit, Unit>() {
            override fun doInBackground() {
                try {
                    ftp = FTPClient()
                    ftp!!.connect(host)

                    if (!FTPReply.isPositiveCompletion(ftp!!.replyCode)) {
                        ftp!!.disconnect()
                        throw Exception("FTP server refused connection")
                    }

                    if (!ftp!!.login(username, password)) {
                        throw Exception("Login failed: ${ftp!!.replyString}")
                    }

                    ftp!!.setFileType(FTPClient.BINARY_FILE_TYPE)
                    isConnected = true
                } catch (e: Exception) {
                    throw e
                }
            }

            override fun done() {
                try {
                    get()
                    onSuccess()
                } catch (e: Exception) {
                    onError("Connection failed: ${e.message}")
                }
            }
        }.execute()
    }

    fun disconnect(onComplete: (() -> Unit)? = null) {
        object : SwingWorker<Unit, Unit>() {
            override fun doInBackground() {
                try {
                    if (ftp?.isConnected == true) {
                        ftp?.logout()
                        ftp?.disconnect()
                    }
                    isConnected = false
                } catch (e: Exception) {
                    println("Error disconnecting: ${e.message}")
                }
            }

            override fun done() {
                onComplete?.invoke()
            }
        }.execute()
    }

    fun listDirectories(
        path: String,
        onSuccess: (List<String>) -> Unit,
        onError: (String) -> Unit
    ) {
        object : SwingWorker<List<String>, Unit>() {
            override fun doInBackground(): List<String> {
                return ftp?.listDirectories(path)
                    ?.filter { it.name != "." && it.name != ".." }
                    ?.map { it.name }
                    ?: emptyList()
            }

            override fun done() {
                try {
                    val directories = get()
                    onSuccess(directories)
                } catch (e: Exception) {
                    onError("Failed to list directories: ${e.message}")
                }
            }
        }.execute()
    }

    fun listFiles(
        path: String,
        onSuccess: (List<FileInfo>) -> Unit,
        onError: (String) -> Unit
    ) {
        object : SwingWorker<List<FileInfo>, Unit>() {
            override fun doInBackground(): List<FileInfo> {
                return ftp?.listFiles(path)
                    ?.filter { it.isFile }
                    ?.sortedByDescending { it.timestamp }
                    ?.map { file ->
                        FileInfo(
                            name = file.name,
                            size = file.size,
                            timestamp = file.timestamp.time.time,
                            path = "$path/${file.name}"
                        )
                    }
                    ?: emptyList()
            }

            override fun done() {
                try {
                    val files = get()
                    onSuccess(files)
                } catch (e: Exception) {
                    onError("Failed to list files: ${e.message}")
                }
            }
        }.execute()
    }

    fun uploadFile(
        localPath: String,
        remotePath: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        object : SwingWorker<Unit, Unit>() {
            override fun doInBackground() {
                try {
                    if (!isConnected) {
                        throw Exception("Not connected to FTP server")
                    }

                    val file = File(localPath)
                    if (!file.exists()) {
                        throw Exception("Local file not found: $localPath")
                    }

                    file.inputStream().use { inputStream ->
                        if (!ftp!!.storeFile(remotePath, inputStream)) {
                            throw Exception("Failed to upload file: ${ftp!!.replyString}")
                        }
                    }
                } catch (e: Exception) {
                    throw e
                }
            }

            override fun done() {
                try {
                    get()
                    onSuccess()
                } catch (e: Exception) {
                    onError("Upload failed: ${e.message}")
                }
            }
        }.execute()
    }

    fun downloadFile(
        remotePath: String,
        localPath: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        object : SwingWorker<Unit, Unit>() {
            override fun doInBackground() {
                try {
                    if (!isConnected) {
                        throw Exception("Not connected to FTP server")
                    }

                    val localFile = File(localPath)
                    localFile.parentFile?.mkdirs()

                    localFile.outputStream().use { outputStream ->
                        if (!ftp!!.retrieveFile(remotePath, outputStream)) {
                            throw Exception("Failed to download file: ${ftp!!.replyString}")
                        }
                    }
                } catch (e: Exception) {
                    throw e
                }
            }

            override fun done() {
                try {
                    get()
                    onSuccess()
                } catch (e: Exception) {
                    onError("Download failed: ${e.message}")
                }
            }
        }.execute()
    }

    fun deleteFile(
        remotePath: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        object : SwingWorker<Unit, Unit>() {
            override fun doInBackground() {
                try {
                    if (!isConnected) {
                        throw Exception("Not connected to FTP server")
                    }

                    if (!ftp!!.deleteFile(remotePath)) {
                        throw Exception("Failed to delete file: ${ftp!!.replyString}")
                    }
                } catch (e: Exception) {
                    throw e
                }
            }

            override fun done() {
                try {
                    get()
                    onSuccess()
                } catch (e: Exception) {
                    onError("Delete failed: ${e.message}")
                }
            }
        }.execute()
    }

    fun isConnectedToServer(): Boolean = isConnected
}

data class FileInfo(
    val name: String,
    val size: Long,
    val timestamp: Long,
    val path: String
)