add_executable(inference
	"activation.h"
	"linear.c"
	"linear.h"
	"main.cu"
	"rand.h"
)

set_property(TARGET inference PROPERTY C_STANDARD 11)
