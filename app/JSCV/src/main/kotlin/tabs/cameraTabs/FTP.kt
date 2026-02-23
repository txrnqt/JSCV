package tabs.cameraTabs

import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPConnectionClosedException
import org.apache.commons.net.ftp.FTPReply
import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import javax.swing.*
import kotlin.concurrent.thread

class FTP : JPanel() {
    private val ftpClient = FTPClient()
    private val server = "192.168.1.19"
    private var isConnected = false

    private val tabbedPane = JTabbedPane()
    private val cameraTabs = mutableMapOf<String, CameraTab>()

    init {
        layout = BorderLayout(10, 10)
        border = BorderFactory.createEmptyBorder(10, 10, 10, 10)
        isVisible = true

        // Connection status panel
        add(createStatusPanel(), BorderLayout.NORTH)

        // Main tabbed pane for cameras
        add(tabbedPane, BorderLayout.CENTER)

        // Initialize FTP connection
        connectToFTP()
    }

    private fun createStatusPanel(): JPanel {
        val panel = JPanel(FlowLayout(FlowLayout.LEFT))
        val statusLabel = JLabel("Connected to $server", SwingConstants.LEFT)
        statusLabel.font = statusLabel.font.deriveFont(12f)

        panel.add(statusLabel)

        return panel
    }

    private fun connectToFTP() {
        thread {
            runCatching {
                ftpClient.connect(server)
                val password = System.getenv("FTP_PASSWORD") ?: ""

                if (ftpClient.login("ftp_user", password)) {
                    println("Login success")
                } else {
                    throw Exception("Login failed")
                }

                println("Connected to $server")
                println(ftpClient.replyString)

                if (!FTPReply.isPositiveCompletion(ftpClient.replyCode)) {
                    println("FTP refused to connect")
                    throw FTPConnectionClosedException("FTP refused to connect")
                }

                isConnected = true
                SwingUtilities.invokeLater {
                    loadCameras()
                }
            }.onFailure { e ->
                e.printStackTrace()
                isConnected = false
                SwingUtilities.invokeLater {
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
        thread {
            runCatching {
                if (!isConnected) return@runCatching

                tabbedPane.removeAll()
                cameraTabs.clear()

                val rootFiles = ftpClient.listFiles("/")
                val directories = ftpClient.listDirectories("/")

                println("${rootFiles.size} files found")
                println("Directories: ${directories.map { it.name }}")

                directories.forEach { dir ->
                    val cameraName = dir.name
                    val cameraTab = CameraTab(cameraName, ftpClient)
                    cameraTabs[cameraName] = cameraTab

                    SwingUtilities.invokeLater {
                        tabbedPane.addTab(cameraName, cameraTab)
                    }
                }

                if (cameraTabs.isEmpty()) {
                    SwingUtilities.invokeLater {
                        tabbedPane.addTab("No Cameras", JLabel("No camera directories found"))
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

    fun disconnect() {
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

    init {
        layout = BorderLayout(10, 10)
        border = BorderFactory.createEmptyBorder(10, 10, 10, 10)

        // Camera header
        add(JLabel("Camera: $cameraName", SwingConstants.CENTER), BorderLayout.NORTH)

        // File list with scroll pane
        fileList.selectionMode = ListSelectionModel.SINGLE_SELECTION
        val scrollPane = JScrollPane(fileList)
        scrollPane.preferredSize = Dimension(400, 300)
        add(scrollPane, BorderLayout.CENTER)

        // Action buttons
        add(createButtonPanel(), BorderLayout.SOUTH)

        // Load files
        loadFilesForCamera()
    }

    private fun createButtonPanel(): JPanel {
        val panel = JPanel(FlowLayout(FlowLayout.CENTER, 10, 5))

        val downloadButton = JButton("Download Selected")
        downloadButton.addActionListener {
            val selectedFile = fileList.selectedValue
            if (selectedFile != null) {
                downloadFile(selectedFile)
            } else {
                JOptionPane.showMessageDialog(this, "Please select a file", "Selection Required", JOptionPane.WARNING_MESSAGE)
            }
        }

        val refreshButton = JButton("Refresh")
        refreshButton.addActionListener {
            loadFilesForCamera()
        }

        val deleteButton = JButton("Delete Selected")
        deleteButton.addActionListener {
            val selectedFile = fileList.selectedValue
            if (selectedFile != null) {
                deleteFile(selectedFile)
            }
        }

        panel.add(downloadButton)
        panel.add(refreshButton)
        panel.add(deleteButton)

        return panel
    }

    private fun loadFilesForCamera() {
        thread {
            runCatching {
                fileListModel.clear()
                val files = ftpClient.listFiles("/$cameraName")

                val fileNames = files
                    .sortedByDescending { it.timestamp }
                    .map { "${it.name} (${it.timestamp.time})" }

                SwingUtilities.invokeLater {
                    fileNames.forEach { fileListModel.addElement(it) }
                }

                println("Loaded ${files.size} files for camera: $cameraName")
            }.onFailure { e ->
                e.printStackTrace()
                SwingUtilities.invokeLater {
                    JOptionPane.showMessageDialog(
                        this,
                        "Failed to load files: ${e.message}",
                        "Load Error",
                        JOptionPane.ERROR_MESSAGE
                    )
                }
            }
        }
    }

    private fun downloadFile(fileName: String) {
        val downloadDir = System.getProperty("user.home") + File.separator + "Downloads"
        Files.createDirectories(Paths.get(downloadDir))

        thread {
            runCatching {
                val cleanFileName = fileName.substringBefore(" (")
                val localFile = File(downloadDir, "$cameraName-$cleanFileName")

                localFile.outputStream().use { out ->
                    if (ftpClient.retrieveFile("/$cameraName/$cleanFileName", out)) {
                        println("Downloaded: $cleanFileName to ${localFile.absolutePath}")
                        SwingUtilities.invokeLater {
                            JOptionPane.showMessageDialog(
                                this,
                                "File downloaded to:\n${localFile.absolutePath}",
                                "Download Complete",
                                JOptionPane.INFORMATION_MESSAGE
                            )
                        }
                    } else {
                        throw Exception("Failed to download file from FTP")
                    }
                }
            }.onFailure { e ->
                e.printStackTrace()
                SwingUtilities.invokeLater {
                    JOptionPane.showMessageDialog(
                        this,
                        "Download failed: ${e.message}",
                        "Download Error",
                        JOptionPane.ERROR_MESSAGE
                    )
                }
            }
        }
    }

    private fun deleteFile(fileName: String) {
        val confirm = JOptionPane.showConfirmDialog(
            this,
            "Delete file: $fileName?",
            "Confirm Delete",
            JOptionPane.YES_NO_OPTION
        )

        if (confirm == JOptionPane.YES_OPTION) {
            thread {
                runCatching {
                    val cleanFileName = fileName.substringBefore(" (")
                    if (ftpClient.deleteFile("/$cameraName/$cleanFileName")) {
                        println("Deleted: $cleanFileName")
                        SwingUtilities.invokeLater {
                            loadFilesForCamera()
                            JOptionPane.showMessageDialog(
                                this,
                                "File deleted successfully",
                                "Success",
                                JOptionPane.INFORMATION_MESSAGE
                            )
                        }
                    } else {
                        throw Exception("Failed to delete file on FTP server")
                    }
                }.onFailure { e ->
                    e.printStackTrace()
                    SwingUtilities.invokeLater {
                        JOptionPane.showMessageDialog(
                            this,
                            "Delete failed: ${e.message}",
                            "Delete Error",
                            JOptionPane.ERROR_MESSAGE
                        )
                    }
                }
            }
        }
    }
}