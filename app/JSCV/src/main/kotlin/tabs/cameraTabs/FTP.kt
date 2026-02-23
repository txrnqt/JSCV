package tabs.cameraTabs

import java.awt.BorderLayout
import javax.swing.*

class FTP : JPanel() {
    init {
        layout = BorderLayout()
        add(JLabel("content", SwingConstants.CENTER), BorderLayout.CENTER)
        isVisible = true
    }
}