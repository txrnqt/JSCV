import com.squareup.javapoet.*
import javax.lang.model.element.Modifier
import java.io.File

fun JavaGen(isFiducial: Boolean = false, isObject: Boolean false) {
    
    val fields = listOf(
        "name" to String::class.java,
        "age" to Int::class.javaPrimitiveType
    )

    val classBuilder = TypeSpec.classBuilder("Person")
        .addModifiers(Modifier.PUBLIC)

    fields.forEach { (name, type) ->
        val field = FieldSpec.builder(type, name, Modifier.PRIVATE)
            .build()

        classBuilder.addField(field)
    }

    val javaFile = JavaFile.builder("com.example", classBuilder.build())
        .build()

    println(javaFile.toString())
}