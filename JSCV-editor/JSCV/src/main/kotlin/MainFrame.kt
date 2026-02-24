import tabs.cameraTabs.CameraTabs
import tabs.cameraTabs.EditorTabs
import javax.swing.*

class MainFrame : JFrame("IDE Layout") {

    init {
        defaultCloseOperation = EXIT_ON_CLOSE
        setSize(900, 700)

        val editorTabs = EditorTabs()
        editorTabs.tabPlacement = JTabbedPane.BOTTOM
        val tabs = CameraTabs();

        val verticalSplit = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            tabs,
            editorTabs
        )
        verticalSplit.resizeWeight = 0.5

        add(verticalSplit)
    }
}