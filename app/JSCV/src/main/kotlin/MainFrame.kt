import tabs.editorTabs.EditorTabs
import tabs.cameraTabs.CameraTabs
import javax.swing.*

class MainFrame : JFrame("IDE Layout") {

    init {
        defaultCloseOperation = EXIT_ON_CLOSE
        setSize(900, 600)

        val editorTabs = EditorTabs()
        val tabs = CameraTabs();

        val verticalSplit = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            tabs,
            editorTabs
        )
        verticalSplit.resizeWeight = 0.7

        add(verticalSplit)
    }
}