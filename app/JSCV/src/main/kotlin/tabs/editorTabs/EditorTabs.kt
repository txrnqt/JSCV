package tabs.cameraTabs

import java.awt.Font
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTabbedPane
import javax.swing.JTextArea

class EditorTabs : JTabbedPane() {
    init {
        addTab("Fiducial", swapTab("Fiducial"))
        addTab("Object", swapTab("Object"))
    }

    private fun swapTab(tab: String): JPanel {
        return when (tab) {
            "Fiducial" -> {
                FiducialEditor()
            }

            "Object" -> {
                ObjectEditor()
            }
            else -> throw IllegalArgumentException("Unknown tab: $tab")
        }
    }
}