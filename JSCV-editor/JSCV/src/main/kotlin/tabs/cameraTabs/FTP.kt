package tabs.cameraTabs

import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPConnectionClosedException
import org.apache.commons.net.ftp.FTPReply
import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.io.File
import java.nio.file.Files
import javax.swing.*
import kotlin.concurrent.thread

class FTP : JPanel() {
    private val ftpClient = FTPClient()
    private var server: String? = null
    private var isConnected = false

    private val tabbedPane = JTabbedPane()
    private val cameraTabs = mutableMapOf<String, CameraTab>()
    private val statusLabel = JLabel("Not connected", SwingConstants.LEFT)
    private val connectButton = JButton("Connect")

    init {
        layout = BorderLayout(10, 10)
        border = BorderFactory.createEmptyBorder(10, 10, 10, 10)
        isVisible = true

        add(createConnectionPanel(), BorderLayout.NORTH)
        add(tabbedPane, BorderLayout.CENTER)
    }

    private fun createConnectionPanel(): JPanel {
        val panel = JPanel(FlowLayout(FlowLayout.LEFT, 10, 5))

        val ipLabel = JLabel("Server IP:")
        val ipField = JTextField("192.168.1.19", 15)

        val userLabel = JLabel("Username:")
        val userField = JTextField("ftp_user", 10)

        val passLabel = JLabel("Password:")
        val passField = JPasswordField(10)
        passField.text = System.getenv("FTP_PASSWORD") ?: ""

        statusLabel.font = statusLabel.font.deriveFont(12f)

        connectButton.addActionListener {
            if (isConnected) {
                disconnect()
                connectButton.text = "Connect"
                statusLabel.text = "Disconnected"
                tabbedPane.removeAll()
                cameraTabs.clear()
                isConnected = false
            } else {
                server = ipField.text.trim()
                if (server.isNullOrEmpty()) {
                    JOptionPane.showMessageDialog(
                        this,
                        "Please enter a valid IP address",
                        "Input Error",
                        JOptionPane.WARNING_MESSAGE
                    )
                    return@addActionListener
                }

                val username = userField.text.trim()
                val password = String(passField.password)

                connectToFTP(username, password)
                connectButton.isEnabled = false
                ipField.isEnabled = false
                userField.isEnabled = false
                passField.isEnabled = false
            }
        }

        panel.add(ipLabel)
        panel.add(ipField)
        panel.add(userLabel)
        panel.add(userField)
        panel.add(passLabel)
        panel.add(passField)
        panel.add(connectButton)
        panel.add(JSeparator(SwingConstants.VERTICAL).apply { preferredSize = Dimension(1, 25) })
        panel.add(statusLabel)

        return panel
    }

    private fun connectToFTP(username: String, password: String) {
        statusLabel.text = "Connecting to $server..."

        thread(isDaemon = true) {
            runCatching {
                ftpClient.connect(server)

                if (!FTPReply.isPositiveCompletion(ftpClient.replyCode)) {
                    ftpClient.disconnect()
                    throw FTPConnectionClosedException("FTP refused to connect")
                }

                if (!ftpClient.login(username, password)) {
                    throw Exception("Login failed: ${ftpClient.replyString}")
                }

                ftpClient.setFileType(FTPClient.BINARY_FILE_TYPE)
                isConnected = true

                println("Connected to $server")
                SwingUtilities.invokeLater {
                    statusLabel.text = "Connected to $server"
                    connectButton.text = "Disconnect"
                    connectButton.isEnabled = true
                    loadCameras()
                }
            }.onFailure { e ->
                e.printStackTrace()
                isConnected = false
                SwingUtilities.invokeLater {
                    connectButton.isEnabled = true
                    statusLabel.text = "Connection failed"
                    JOptionPane.showMessageDialog(
                        this,
                        "Connection failed: ${e.message}",
                        "FTP Error",
                        JOptionPane.ERROR_MESSAGE
                    )
                }
            }
        }
    }

    private fun loadCameras() {
        thread(isDaemon = true) {
            runCatching {
                if (!isConnected) return@runCatching

                tabbedPane.removeAll()
                cameraTabs.clear()

                val directories = ftpClient.listDirectories("/")
                    .filter { it.name != "." && it.name != ".." }

                println("Found ${directories.size} camera directories")

                if (directories.isEmpty()) {
                    SwingUtilities.invokeLater {
                        tabbedPane.addTab("No Cameras", JLabel("No camera directories found"))
                    }
                    return@runCatching
                }

                directories.forEach { dir ->
                    val cameraName = dir.name
                    val cameraTab = CameraTab(cameraName, ftpClient)
                    cameraTabs[cameraName] = cameraTab

                    SwingUtilities.invokeLater {
                        tabbedPane.addTab(cameraName, cameraTab)
                    }
                }
            }.onFailure { e ->
                e.printStackTrace()
                SwingUtilities.invokeLater {
                    JOptionPane.showMessageDialog(
                        this,
                        "Failed to load cameras: ${e.message}",
                        "Load Error",
                        JOptionPane.ERROR_MESSAGE
                    )
                }
            }
        }
    }

    private fun disconnect() {
        if (ftpClient.isConnected) {
            runCatching {
                ftpClient.logout()
                ftpClient.disconnect()
            }
            isConnected = false
        }
    }
}

class CameraTab(
    private val cameraName: String,
    private val ftpClient: FTPClient
) : JPanel() {
    private val fileListModel = DefaultListModel<String>()
    private val fileList = JList(fileListModel)
    private var isLoading = false

    init {
        layout = BorderLayout(10, 10)
        border = BorderFactory.createEmptyBorder(10, 10, 10, 10)

        add(JLabel("Camera: $cameraName", SwingConstants.CENTER), BorderLayout.NORTH)

        fileList.selectionMode = ListSelectionModel.SINGLE_SELECTION
        val scrollPane = JScrollPane(fileList)
        scrollPane.preferredSize = Dimension(400, 300)
        add(scrollPane, BorderLayout.CENTER)

        add(createButtonPanel(), BorderLayout.SOUTH)

        loadFilesForCamera()
    }

    private fun createButtonPanel(): JPanel {
        val panel = JPanel(FlowLayout(FlowLayout.CENTER, 10, 5))

        val downloadButton = JButton("Download Selected")
        downloadButton.addActionListener {
            val selectedFile = fileList.selectedValue
            when {
                selectedFile == null -> JOptionPane.showMessageDialog(
                    this,
                    "Please select a file",
                    "Selection Required",
                    JOptionPane.WARNING_MESSAGE
                )
                isLoading -> JOptionPane.showMessageDialog(
                    this,
                    "Please wait, operation in progress",
                    "Busy",
                    JOptionPane.WARNING_MESSAGE
                )
                else -> downloadFile(selectedFile)
            }
        }

        val refreshButton = JButton("Refresh")
        refreshButton.addActionListener {
            if (!isLoading) loadFilesForCamera()
        }

        val deleteButton = JButton("Delete Selected")
        deleteButton.addActionListener {
            val selectedFile = fileList.selectedValue
            if (selectedFile != null && !isLoading) {
                deleteFile(selectedFile)
            }
        }

        panel.add(downloadButton)
        panel.add(refreshButton)
        panel.add(deleteButton)

        return panel
    }

    private fun loadFilesForCamera() {
        if (isLoading) return

        thread(isDaemon = true) {
            try {
                isLoading = true
                fileListModel.clear()

                val files = ftpClient.listFiles("/$cameraName")
                    .filter { it.isFile }
                    .sortedByDescending { it.timestamp }

                val fileNames = files.map { file ->
                    val sizeKB = file.size / 1024
                    "${file.name} (${file.timestamp.time}) [$sizeKB KB]"
                }

                SwingUtilities.invokeLater {
                    fileNames.forEach { fileListModel.addElement(it) }
                    println("Loaded ${files.size} files for camera: $cameraName")
                }
            } catch (e: Exception) {
                e.printStackTrace()
                SwingUtilities.invokeLater {
                    JOptionPane.showMessageDialog(
                        this,
                        "Failed to load files: ${e.message}",
                        "Load Error",
                        JOptionPane.ERROR_MESSAGE
                    )
                }
            } finally {
                isLoading = false
            }
        }
    }

    private fun downloadFile(fileName: String) {
        val downloadDir = File(System.getProperty("user.home"), "Downloads")
        Files.createDirectories(downloadDir.toPath())

        thread(isDaemon = true) {
            try {
                isLoading = true
                val cleanFileName = fileName.substringBefore(" (")
                val localFile = File(downloadDir, "$cameraName-$cleanFileName")

                localFile.outputStream().use { out ->
                    if (ftpClient.retrieveFile("/$cameraName/$cleanFileName", out)) {
                        SwingUtilities.invokeLater {
                            JOptionPane.showMessageDialog(
                                this,
                                "Downloaded:\n${localFile.absolutePath}",
                                "Success",
                                JOptionPane.INFORMATION_MESSAGE
                            )
                        }
                    } else {
                        throw Exception("FTP serverWorkers.server rejected download request")
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                SwingUtilities.invokeLater {
                    JOptionPane.showMessageDialog(
                        this,
                        "Download failed: ${e.message}",
                        "Error",
                        JOptionPane.ERROR_MESSAGE
                    )
                }
            } finally {
                isLoading = false
            }
        }
    }

    private fun deleteFile(fileName: String) {
        val confirm = JOptionPane.showConfirmDialog(
            this,
            "Delete: $fileName?",
            "Confirm",
            JOptionPane.YES_NO_OPTION,
            JOptionPane.WARNING_MESSAGE
        )

        if (confirm == JOptionPane.YES_OPTION) {
            thread(isDaemon = true) {
                try {
                    isLoading = true
                    val cleanFileName = fileName.substringBefore(" (")

                    if (ftpClient.deleteFile("/$cameraName/$cleanFileName")) {
                        SwingUtilities.invokeLater {
                            loadFilesForCamera()
                            JOptionPane.showMessageDialog(
                                this,
                                "File deleted",
                                "Success",
                                JOptionPane.INFORMATION_MESSAGE
                            )
                        }
                    } else {
                        throw Exception("FTP serverWorkers.server rejected delete request")
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    SwingUtilities.invokeLater {
                        JOptionPane.showMessageDialog(
                            this,
                            "Delete failed: ${e.message}",
                            "Error",
                            JOptionPane.ERROR_MESSAGE
                        )
                    }
                } finally {
                    isLoading = false
                }
            }
        }
    }
}