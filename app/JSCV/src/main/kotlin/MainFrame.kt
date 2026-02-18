import javax.swing.*
import java.awt.*

class MainFrame : JFrame("IDE Layout") {

    init {
        defaultCloseOperation = EXIT_ON_CLOSE
        setSize(900, 600)

        // --- Sidebar (file explorer) ---
        val sidebar = JPanel()
        sidebar.layout = BoxLayout(sidebar, BoxLayout.Y_AXIS)
        sidebar.add(JButton("File1.kt"))
        sidebar.add(JButton("File2.kt"))
        sidebar.add(JButton("File3.kt"))

        // --- Code editor area ---
        val editor = JTextArea("Code goes here...")
        val editorScroll = JScrollPane(editor)

        // --- Terminal area ---
        val terminal = JTextArea("Terminal output...")
        val terminalScroll = JScrollPane(terminal)

        // Split editor and terminal (top/bottom)
        val verticalSplit = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            editorScroll,
            terminalScroll
        )
        verticalSplit.resizeWeight = 0.7 // editor gets more space

        // Split sidebar and main workspace (left/right)
        val horizontalSplit = JSplitPane(
            JSplitPane.HORIZONTAL_SPLIT,
            sidebar,
            verticalSplit
        )
        horizontalSplit.resizeWeight = 0.2

        add(horizontalSplit)
    }
}