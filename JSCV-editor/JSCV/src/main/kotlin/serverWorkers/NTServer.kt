package serverWorkers

import AprilTagConfig
import CameraConfig
import LoggingConfig
import ObjConfig
import ServerConfig
import edu.wpi.first.networktables.NetworkTableInstance
import edu.wpi.first.networktables.NetworkTable
import edu.wpi.first.networktables.StringPublisher
import edu.wpi.first.networktables.IntegerPublisher
import edu.wpi.first.networktables.DoublePublisher
import edu.wpi.first.networktables.BooleanPublisher

/**
 * Publishes configuration values to NetworkTables for the vision system.
 */
class server(
    private val serverConfig: ServerConfig,
    private val cameraConfig: CameraConfig,
    private val loggingConfig: LoggingConfig,
    private val apriltagConfig: AprilTagConfig,
    private val objConfig: ObjConfig
) {
    private var initComplete = false

    // Camera publishers
    private lateinit var cameraIdPub: StringPublisher
    private lateinit var cameraResolutionWidthPub: IntegerPublisher
    private lateinit var cameraResolutionHeightPub: IntegerPublisher
    private lateinit var cameraAutoExposurePub: IntegerPublisher
    private lateinit var cameraExposurePub: DoublePublisher
    private lateinit var cameraGainPub: DoublePublisher
    private lateinit var cameraDenoisePub: DoublePublisher

    // Logging publishers
    private lateinit var isRecordingPub: BooleanPublisher
    private lateinit var timeStampPub: DoublePublisher
    private lateinit var loggingLocationPub: StringPublisher
    private lateinit var eventNamePub: StringPublisher
    private lateinit var matchTypePub: StringPublisher
    private lateinit var matchNumberPub: IntegerPublisher

    // AprilTag publishers
    private lateinit var fiducialLayoutPub: DoublePublisher
    private lateinit var fiducialSizePub: DoublePublisher

    // Object detection publishers
    private lateinit var backendTypeObjPub: StringPublisher

    fun initialize() {
        if (!initComplete) {
            initializePublishers()
            initComplete = true
        }
    }

    private fun initializePublishers() {
        try {
            val ntInstance = NetworkTableInstance.getDefault()
            val ntTable: NetworkTable = ntInstance.getTable(
                "/${serverConfig.device_id}/config"
            )

            // Camera publishers
            cameraIdPub = ntTable.getStringTopic("camera_id").publish()
            cameraResolutionWidthPub = ntTable.getIntegerTopic("camera_resolution_width").publish()
            cameraResolutionHeightPub = ntTable.getIntegerTopic("camera_resolution_height").publish()
            cameraAutoExposurePub = ntTable.getIntegerTopic("camera_auto_exposure").publish()
            cameraExposurePub = ntTable.getDoubleTopic("camera_exposure").publish()
            cameraGainPub = ntTable.getDoubleTopic("camera_gain").publish()
            cameraDenoisePub = ntTable.getDoubleTopic("camera_denoise").publish()

            // Logging publishers
            isRecordingPub = ntTable.getBooleanTopic("is_recording").publish()
            timeStampPub = ntTable.getDoubleTopic("time_stamp").publish()
            loggingLocationPub = ntTable.getStringTopic("logging_location").publish()
            eventNamePub = ntTable.getStringTopic("event_name").publish()
            matchTypePub = ntTable.getStringTopic("match_type").publish()
            matchNumberPub = ntTable.getIntegerTopic("match_number").publish()

            // AprilTag publishers
            fiducialLayoutPub = ntTable.getDoubleTopic("fiducial_layout").publish()
            fiducialSizePub = ntTable.getDoubleTopic("fiducial_size").publish()

            // Object detection publishers
            backendTypeObjPub = ntTable.getStringTopic("backend_type_object").publish()

            println("NetworkTables publishers initialized successfully")
        } catch (e: Exception) {
            println("Error initializing NetworkTables publishers: ${e.message}")
            e.printStackTrace()
        }
    }

    fun publishAllConfig() {
        if (!initComplete) initialize()

        publishCameraConfig()
        publishLoggingConfig()
        publishAprilTagConfig()
        publishObjConfig()
    }

    fun publishCameraConfig() {
        if (!initComplete) initialize()

        try {
            cameraIdPub.set(cameraConfig.camera_id)
            cameraResolutionWidthPub.set(cameraConfig.camera_resolution_width.toLong())
            cameraResolutionHeightPub.set(cameraConfig.camera_resolution_height.toLong())
            cameraAutoExposurePub.set(cameraConfig.camera_auto_exposure.toLong())
            cameraExposurePub.set(cameraConfig.camera_exposure)
            cameraGainPub.set(cameraConfig.camera_gain)
            cameraDenoisePub.set(cameraConfig.camera_denoise)
        } catch (e: Exception) {
            println("Error publishing camera config: ${e.message}")
        }
    }

    fun publishLoggingConfig() {
        if (!initComplete) initialize()

        try {
            isRecordingPub.set(loggingConfig.is_recording)
            timeStampPub.set(loggingConfig.time_stamp.toDouble())
            loggingLocationPub.set(loggingConfig.logging_location)
            eventNamePub.set(loggingConfig.event_name)
            matchTypePub.set(loggingConfig.match_type)
            matchNumberPub.set(loggingConfig.match_number.toLong())
        } catch (e: Exception) {
            println("Error publishing logging config: ${e.message}")
        }
    }

    fun publishAprilTagConfig() {
        if (!initComplete) initialize()

        try {
            if (apriltagConfig.fiducial_layout != null) {
                fiducialLayoutPub.set(apriltagConfig.fiducial_layout!!)
            }
            fiducialSizePub.set(apriltagConfig.fiducial_size)
        } catch (e: Exception) {
            println("Error publishing AprilTag config: ${e.message}")
        }
    }

    fun publishObjConfig() {
        if (!initComplete) initialize()

        try {
            backendTypeObjPub.set(objConfig.backend)
        } catch (e: Exception) {
            println("Error publishing object detection config: ${e.message}")
        }
    }

    // Individual setters for specific values
    fun setCameraId(id: String) {
        cameraConfig.camera_id = id
        cameraIdPub.set(id)
    }

    fun setCameraResolution(width: Int, height: Int) {
        cameraConfig.camera_resolution_width = width
        cameraConfig.camera_resolution_height = height
        cameraResolutionWidthPub.set(width.toLong())
        cameraResolutionHeightPub.set(height.toLong())
    }

    fun setCameraExposure(exposure: Double) {
        cameraConfig.camera_exposure = exposure
        cameraExposurePub.set(exposure)
    }

    fun setCameraGain(gain: Double) {
        cameraConfig.camera_gain = gain
        cameraGainPub.set(gain)
    }

    fun setCameraDenoise(denoise: Double) {
        cameraConfig.camera_denoise = denoise
        cameraDenoisePub.set(denoise)
    }

    fun setRecording(isRecording: Boolean) {
        loggingConfig.is_recording = isRecording
        isRecordingPub.set(isRecording)
    }

    fun setEventName(eventName: String) {
        loggingConfig.event_name = eventName
        eventNamePub.set(eventName)
    }

    fun setMatchInfo(matchType: String, matchNumber: Int) {
        loggingConfig.match_type = matchType
        loggingConfig.match_number = matchNumber
        matchTypePub.set(matchType)
        matchNumberPub.set(matchNumber.toLong())
    }

    fun setFiducialSize(size: Double) {
        apriltagConfig.fiducial_size = size
        fiducialSizePub.set(size)
    }

    fun setBackend(backend: String) {
        objConfig.backend = backend
        backendTypeObjPub.set(backend)
    }
}