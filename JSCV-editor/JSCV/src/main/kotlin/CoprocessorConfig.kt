/**
 * Configuration data classes for the vision system
 */

data class ServerConfig(
    var device_id: String = "",
    var server_ip: String = "",
    var april_tag_port: Int = 0,
    var obj_port: Int = 0,
    var april_tag_tracking: Boolean = false,
    var obj_tracking: Boolean = false
)

data class CameraConfig(
    var camera_id: String = "",
    var camera_resolution_width: Int = 640,
    var camera_resolution_height: Int = 480,
    var camera_auto_exposure: Int = 1,
    var camera_exposure: Double = 0.0,
    var camera_gain: Double = 0.0,
    var camera_denoise: Double = 0.0,
    var camera_matrix: Array<DoubleArray>? = null,
    var distortion_coefficients: Array<DoubleArray>? = null,
    var has_config: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CameraConfig) return false
        if (camera_id != other.camera_id) return false
        if (camera_resolution_width != other.camera_resolution_width) return false
        if (camera_resolution_height != other.camera_resolution_height) return false
        if (camera_auto_exposure != other.camera_auto_exposure) return false
        if (camera_exposure != other.camera_exposure) return false
        if (camera_gain != other.camera_gain) return false
        if (camera_denoise != other.camera_denoise) return false
        if (has_config != other.has_config) return false
        return true
    }

    override fun hashCode(): Int {
        var result = camera_id.hashCode()
        result = 31 * result + camera_resolution_width
        result = 31 * result + camera_resolution_height
        result = 31 * result + camera_auto_exposure
        result = 31 * result + camera_exposure.hashCode()
        result = 31 * result + camera_gain.hashCode()
        result = 31 * result + camera_denoise.hashCode()
        result = 31 * result + has_config.hashCode()
        return result
    }
}

data class LoggingConfig(
    var is_recording: Boolean = false,
    var time_stamp: Long = 0L,
    var logging_location: String = "",
    var event_name: String = "",
    var match_type: String = "",
    var match_number: Int = 0
)

data class AprilTagConfig(
    var fiducial_layout: Double? = null,
    var fiducial_size: Double = 0.0
)

data class ObjConfig(
    var backend: String = ""
)