package tabs.cameraTabs

import java.awt.Font
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTabbedPane
import javax.swing.JTextArea

class CameraTabs : JTabbedPane() {
    init {
        addTab("Camera View", swapTab("Camera View"))
        addTab("Configuration", swapTab("Configuration"))
        addTab("FTP", swapTab("FTP"))
        addTab("Console", swapTab("Console"))
    }

    private fun swapTab(tab: String): JPanel {
        return when (tab) {
            "Camera View" -> {
                CameraView()
            }

            "Configuration" -> {
                Configuration()
            }

            "FTP" -> {
                FTP()
            }

            "Console" -> {
                Console()
            }

            else -> throw IllegalArgumentException("Unknown tab: $tab")
        }
    }
}