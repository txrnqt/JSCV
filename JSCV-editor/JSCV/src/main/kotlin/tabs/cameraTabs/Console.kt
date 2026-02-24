package tabs.cameraTabs

import serverWorkers.SshWorker
import javax.swing.*
import java.awt.*

class Console : JPanel() {
    private val sshManager = SshWorker()
    private val outputArea = JTextArea(15, 50)
    private val connectButton = JButton("Connect")

    private val hostField = JTextField("localhost", 15).apply {
        preferredSize = Dimension(150, 25)
    }
    private val userField = JTextField("user", 15).apply {
        preferredSize = Dimension(150, 25)
    }
    private val passField = JPasswordField(15).apply {
        preferredSize = Dimension(150, 25)
    }

    init {
        layout = BorderLayout()

        hostField.addFocusListener(object : java.awt.event.FocusAdapter() {
            override fun focusGained(e: java.awt.event.FocusEvent?) {
                if (hostField.text == "localhost") {
                    hostField.text = ""
                }
            }

            override fun focusLost(e: java.awt.event.FocusEvent?) {
                if (hostField.text.isEmpty()) {
                    hostField.text = "localhost"
                }
            }
        })

        userField.addFocusListener(object : java.awt.event.FocusAdapter() {
            override fun focusGained(e: java.awt.event.FocusEvent?) {
                if (userField.text == "user") {
                    userField.text = ""
                }
            }

            override fun focusLost(e: java.awt.event.FocusEvent?) {
                if (userField.text.isEmpty()) {
                    userField.text = "user"
                }
            }
        })

        passField.addFocusListener(object : java.awt.event.FocusAdapter() {
            override fun focusGained(e: java.awt.event.FocusEvent?) {
                if (String(passField.password) == "password") {
                    passField.text = ""
                }
            }

            override fun focusLost(e: java.awt.event.FocusEvent?) {
                if (passField.password.isEmpty()) {
                    passField.text = "password"
                }
            }
        })

        val connectionPanel = JPanel(GridBagLayout()).apply {
            border = BorderFactory.createTitledBorder("SSH Connection")
            val gbc = GridBagConstraints().apply {
                insets = Insets(5, 5, 5, 5)
                fill = GridBagConstraints.BOTH
                ipadx = 10
                ipady = 5
            }

            gbc.gridx = 0
            gbc.gridy = 0
            gbc.weightx = 0.0
            add(JLabel("Host:"), gbc)
            gbc.gridx = 1
            gbc.weightx = 1.0
            add(hostField, gbc)

            gbc.gridx = 2
            gbc.weightx = 0.0
            add(JLabel("User:"), gbc)
            gbc.gridx = 3
            gbc.weightx = 1.0
            add(userField, gbc)

            gbc.gridx = 4
            gbc.weightx = 0.0
            add(JLabel("Password:"), gbc)
            gbc.gridx = 5
            gbc.weightx = 1.0
            add(passField, gbc)

            gbc.gridx = 6
            gbc.weightx = 0.0
            connectButton.addActionListener {
                connectButton.isEnabled = false
                sshManager.connect(
                    hostField.text,
                    userField.text,
                    String(passField.password),
                    onSuccess = {
                        outputArea.append("Connected successfully\n")
                        outputArea.isEditable = true
                        outputArea.requestFocus()
                        connectButton.isEnabled = true
                    },
                    onError = { error ->
                        outputArea.append("Error: $error\n")
                        connectButton.isEnabled = true
                    }
                )
            }
            add(connectButton, gbc)
        }

        outputArea.isEditable = false
        outputArea.font = Font("Monospaced", Font.PLAIN, 12)

        outputArea.addKeyListener(object : java.awt.event.KeyAdapter() {
            override fun keyPressed(e: java.awt.event.KeyEvent?) {
                if (e?.keyCode == java.awt.event.KeyEvent.VK_ENTER) {
                    e.consume()

                    val text = outputArea.text
                    val lastNewline = text.lastIndexOf('\n')
                    val command = if (lastNewline == -1) {
                        text
                    } else {
                        text.substring(lastNewline + 1)
                    }

                    if (command.isNotEmpty() && !command.startsWith("$ ")) {
                        outputArea.append("\n")

                        sshManager.executeCommand(
                            command,
                            onResult = { result ->
                                outputArea.append(result)
                                if (!result.endsWith("\n")) {
                                    outputArea.append("\n")
                                }
                            },
                            onError = { error ->
                                outputArea.append("Error: $error\n")
                            }
                        )
                    }
                }
            }
        })

        val scrollPane = JScrollPane(outputArea)

        add(connectionPanel, BorderLayout.NORTH)
        add(scrollPane, BorderLayout.CENTER)
    }

    override fun removeNotify() {
        super.removeNotify()
    }
}