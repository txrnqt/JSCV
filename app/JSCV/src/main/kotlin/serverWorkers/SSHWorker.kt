package serverWorkers

import org.apache.sshd.client.SshClient
import org.apache.sshd.client.session.ClientSession
import org.apache.sshd.common.util.io.IoUtils
import java.io.ByteArrayOutputStream
import javax.swing.SwingWorker

class SshWorker {
    private var client: SshClient? = null
    private var session: ClientSession? = null

    fun connect(host: String, user: String, password: String, onSuccess: () -> Unit, onError: (String) -> Unit) {
        object : SwingWorker<Unit, Unit>() {
            override fun doInBackground() {
                try {
                    client = SshClient.setUpDefaultClient()
                    client!!.start()

                    session = client!!.connect(user, host, 22).verify().session
                    session!!.addPasswordIdentity(password)
                    session!!.auth().verify()
                } catch (e: Exception) {
                    throw e
                }
            }

            override fun done() {
                try {
                    get()
                    onSuccess()
                } catch (e: Exception) {
                    onError("Connection failed: ${e.message}")
                }
            }
        }.execute()
    }

    fun executeCommand(command: String, onResult: (String) -> Unit, onError: (String) -> Unit) {
        object : SwingWorker<String, Unit>() {
            override fun doInBackground(): String {
                return try {
                    val channel = session!!.createExecChannel(command)
                    val output = java.io.ByteArrayOutputStream()
                    channel.out = output
                    channel.open().verify()

                    while (channel.isOpen) {
                        Thread.sleep(100)
                    }

                    output.toString()
                } catch (e: Exception) {
                    throw e
                }
            }

            override fun done() {
                try {
                    onResult(get())
                } catch (e: Exception) {
                    onError("Command failed: ${e.message}")
                }
            }
        }.execute()
    }
}