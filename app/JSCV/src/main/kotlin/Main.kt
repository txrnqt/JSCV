import javax.swing.*
import java.awt.*

fun main() {
    SwingUtilities.invokeLater {
        val frame = JFrame("My App").apply {
            defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            setSize(300, 200)
            setLocationRelativeTo(null)
            layout = BorderLayout()
        }

        var count = 0
        val label = JLabel("Count: 0", SwingConstants.CENTER)
        label.font = Font("Arial", Font.BOLD, 24)

        val panel = JPanel().apply {
            add(JButton("-").also { btn ->
                btn.addActionListener { label.text = "Count: ${--count}" }
            })
            add(JButton("+").also { btn ->
                btn.addActionListener { label.text = "Count: ${++count}" }
            })
        }

        frame.add(label, BorderLayout.CENTER)
        frame.add(panel, BorderLayout.SOUTH)
        frame.isVisible = true
    }
}