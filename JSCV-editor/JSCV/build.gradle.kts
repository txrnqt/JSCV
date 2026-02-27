plugins {
    kotlin("jvm") version "2.3.0"
}

group = "org"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("commons-net:commons-net:3.9.0")
    implementation("org.apache.sshd:sshd-core:2.13.0")
    implementation("com.squareup:javapoet:1.13.0")
}

kotlin {
    jvmToolchain(24)
}

tasks.test {
    useJUnitPlatform()
}