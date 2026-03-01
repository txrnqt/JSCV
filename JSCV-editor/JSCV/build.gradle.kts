plugins {
    kotlin("jvm") version "2.3.0"
}

group = "org"
version = "1.0-SNAPSHOT"

repositories { mavenCentral()
    maven {
        url = uri("https://frcmaven.wpi.edu/artifactory/release/")
    }
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("commons-net:commons-net:3.9.0")
    implementation("org.apache.sshd:sshd-core:2.13.0")
    implementation("com.squareup:javapoet:1.13.0")

    implementation("edu.wpi.first.ntcore:ntcore-java:2024.3.2")
    implementation("edu.wpi.first.wpiutil:wpiutil-java:2024.3.2")
}

kotlin {
    jvmToolchain(24)
}

tasks.test {
    useJUnitPlatform()
}