#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>

/* ── Matrix math ─────────────────────────────────────────────────────────────
   OpenGL uses column-major 4x4 matrices.
   Index layout: m[col * 4 + row]
────────────────────────────────────────────────────────────────────────────── */

typedef float mat4[16];

static void mat4_identity(mat4 m)
{
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4_multiply(mat4 out, const mat4 a, const mat4 b)
{
    mat4 tmp;
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            tmp[col*4 + row] = 0.0f;
            for (int k = 0; k < 4; k++)
                tmp[col*4 + row] += a[k*4 + row] * b[col*4 + k];
        }
    for (int i = 0; i < 16; i++) out[i] = tmp[i];
}

static void mat4_rotate_y(mat4 m, float angle)
{
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0]  =  c;
    m[2]  = -s;
    m[8]  =  s;
    m[10] =  c;
}

static void mat4_rotate_x(mat4 m, float angle)
{
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[5]  =  c;
    m[6]  =  s;
    m[9]  = -s;
    m[10] =  c;
}

static void mat4_rotate_z(mat4 m, float angle)
{
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0]  =  c;
    m[1]  =  s;
    m[4]  = -s;
    m[5]  =  c;
}

static void mat4_translate(mat4 m, float x, float y, float z)
{
    mat4_identity(m);
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

static void mat4_perspective(mat4 m, float fov, float aspect, float near_z, float far_z)
{
    float f = 1.0f / tanf(fov * 0.5f);
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (far_z + near_z) / (near_z - far_z);
    m[11] = -1.0f;
    m[14] = (2.0f * far_z * near_z) / (near_z - far_z);
}

/* ── Serial / IMU input ──────────────────────────────────────────────────────
   Reads roll, pitch, yaw from the Arduino over USB serial.
   Non-blocking — if no data is available the renderer keeps running normally.
────────────────────────────────────────────────────────────────────────────── */

#define SERIAL_PORT  "/tmp/imu_pipe"
#define DEG_TO_RAD   0.01745329251994f
#define SMOOTH_TAU       0.08f   /* smoothing time constant in seconds — increase for more lag */

static int  serial_fd      = -1;
static char serial_buf[128];
static int  serial_buf_len = 0;

static int serial_open(const char *path)
{
    /* Create the named pipe if it doesn't exist yet */
    mkfifo(path, 0666);
    int fd = open(path, O_RDONLY | O_NONBLOCK);
    return fd;
}

static int serial_read_angles(float *r, float *p, float *y)
{
    if (serial_fd < 0) return 0;

    char c;
    int  got = 0;
    while (read(serial_fd, &c, 1) == 1) {
        if (c == '\n') {
            serial_buf[serial_buf_len] = '\0';
            if (sscanf(serial_buf, "%f,%f,%f", r, p, y) == 3)
                got = 1;
            serial_buf_len = 0;
        } else if (c != '\r' && serial_buf_len < (int)sizeof(serial_buf) - 1) {
            serial_buf[serial_buf_len++] = c;
        }
    }
    return got;
}

/* ── Input state ─────────────────────────────────────────────────────────────*/

static float yaw   = 0.0f;
static float pitch = 0.0f;
static float roll  = 0.0f;

static int    mouse_dragging = 0;
static double last_mouse_x   = 0.0;
static double last_mouse_y   = 0.0;

#define MOUSE_SENSITIVITY  0.005f

/* ── GLFW callbacks ──────────────────────────────────────────────────────────*/

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouse_dragging = 1;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            mouse_dragging = 0;
        }
    }
}

/* Mouse drag is only active when no IMU is connected */
void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
    (void)window;
    if (!mouse_dragging || serial_fd >= 0) return;

    float dx = (float)(xpos - last_mouse_x);
    float dy = (float)(ypos - last_mouse_y);
    last_mouse_x = xpos;
    last_mouse_y = ypos;

    yaw   += dx * MOUSE_SENSITIVITY;
    pitch += dy * MOUSE_SENSITIVITY;
}

/* ── Shader source ───────────────────────────────────────────────────────────*/

const char *vertexShaderSource =
    "#version 410 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aColor;\n"
    "layout (location = 2) in vec3 aNormal;\n"
    "out vec3 fragColor;\n"
    "out vec3 fragNormal;\n"
    "out vec3 fragPos;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "void main() {\n"
    "    vec4 worldPos  = model * vec4(aPos, 1.0);\n"
    "    gl_Position    = projection * view * worldPos;\n"
    "    fragPos        = vec3(worldPos);\n"
    "    fragNormal     = mat3(model) * aNormal;\n"
    "    fragColor      = aColor;\n"
    "}\0";

const char *fragmentShaderSource =
    "#version 410 core\n"
    "in vec3 fragColor;\n"
    "in vec3 fragNormal;\n"
    "in vec3 fragPos;\n"
    "out vec4 FragColor;\n"
    "uniform vec3 cameraPos;\n"
    "uniform int  lightingEnabled;\n"
    "void main() {\n"
    "    if (lightingEnabled == 0) {\n"
    "        FragColor = vec4(fragColor, 1.0);\n"
    "        return;\n"
    "    }\n"
    "    vec3  lightDir    = normalize(vec3(1.0, 2.0, 1.5));\n"
    "    vec3  norm        = normalize(fragNormal);\n"
    "    vec3  viewDir     = normalize(cameraPos - fragPos);\n"
    "    vec3  reflectDir  = reflect(-lightDir, norm);\n"
    "    float ambient     = 0.15;\n"
    "    float diff        = max(dot(norm, lightDir), 0.0);\n"
    "    float spec        = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);\n"
    "    vec3  result      = (ambient + diff) * fragColor\n"
    "                      + 0.5 * spec * vec3(1.0);\n"
    "    FragColor = vec4(result, 1.0);\n"
    "}\0";

unsigned int compile_shader(unsigned int type, const char *source)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
    }
    return shader;
}

int main(void)
{
    /* ── Init GLFW ─────────────────────────────────────────────────────────── */
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "Pyramid — Direct IMU", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n");
        return -1;
    }

    printf("OpenGL version: %s\n", glGetString(GL_VERSION));

    serial_fd = serial_open(SERIAL_PORT);
    if (serial_fd < 0)
        printf("Pipe: failed to open %s — mouse drag active\n", SERIAL_PORT);
    else
        printf("Pipe: listening on %s (start ble_relay.py to stream)\n", SERIAL_PORT);

    glEnable(GL_DEPTH_TEST);

    /* ── Pyramid vertex data ─────────────────────────────────────────────────
       9 floats per vertex: x,y,z  r,g,b  nx,ny,nz
    ────────────────────────────────────────────────────────────────────────── */
    float vertices[] = {
        /* Front face — orange */
         0.0f,  0.7f,  0.0f,   1.0f, 0.5f, 0.0f,   0.0f,    0.4472f,  0.8944f,
        -0.5f, -0.3f,  0.5f,   1.0f, 0.5f, 0.0f,   0.0f,    0.4472f,  0.8944f,
         0.5f, -0.3f,  0.5f,   1.0f, 0.5f, 0.0f,   0.0f,    0.4472f,  0.8944f,

        /* Right face — red */
         0.0f,  0.7f,  0.0f,   0.8f, 0.1f, 0.1f,   0.8944f, 0.4472f,  0.0f,
         0.5f, -0.3f,  0.5f,   0.8f, 0.1f, 0.1f,   0.8944f, 0.4472f,  0.0f,
         0.5f, -0.3f, -0.5f,   0.8f, 0.1f, 0.1f,   0.8944f, 0.4472f,  0.0f,

        /* Back face — green */
         0.0f,  0.7f,  0.0f,   0.1f, 0.7f, 0.2f,   0.0f,    0.4472f, -0.8944f,
         0.5f, -0.3f, -0.5f,   0.1f, 0.7f, 0.2f,   0.0f,    0.4472f, -0.8944f,
        -0.5f, -0.3f, -0.5f,   0.1f, 0.7f, 0.2f,   0.0f,    0.4472f, -0.8944f,

        /* Left face — blue */
         0.0f,  0.7f,  0.0f,   0.2f, 0.3f, 0.9f,  -0.8944f, 0.4472f,  0.0f,
        -0.5f, -0.3f, -0.5f,   0.2f, 0.3f, 0.9f,  -0.8944f, 0.4472f,  0.0f,
        -0.5f, -0.3f,  0.5f,   0.2f, 0.3f, 0.9f,  -0.8944f, 0.4472f,  0.0f,

        /* Base triangle 1 — dark grey */
        -0.5f, -0.3f,  0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
         0.5f, -0.3f,  0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
         0.5f, -0.3f, -0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,

        /* Base triangle 2 — dark grey */
        -0.5f, -0.3f,  0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
         0.5f, -0.3f, -0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
        -0.5f, -0.3f, -0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
    };

    /* ── VAO / VBO ─────────────────────────────────────────────────────────── */
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    /* ── Build shader program ──────────────────────────────────────────────── */
    unsigned int vertexShader   = compile_shader(GL_VERTEX_SHADER,   vertexShaderSource);
    unsigned int fragmentShader = compile_shader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char log[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        fprintf(stderr, "Shader link error: %s\n", log);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    int modelLoc           = glGetUniformLocation(shaderProgram, "model");
    int viewLoc            = glGetUniformLocation(shaderProgram, "view");
    int projectionLoc      = glGetUniformLocation(shaderProgram, "projection");
    int cameraPosLoc       = glGetUniformLocation(shaderProgram, "cameraPos");
    int lightingEnabledLoc = glGetUniformLocation(shaderProgram, "lightingEnabled");

    mat4 projection;
    mat4_perspective(projection, 45.0f * DEG_TO_RAD, 800.0f / 600.0f, 0.1f, 100.0f);

    mat4 view;
    mat4_translate(view, 0.0f, 0.0f, -2.5f);

    glUseProgram(shaderProgram);
    glUniform3f(cameraPosLoc, 0.0f, 0.0f, 2.5f);

    /* ── IMU state ─────────────────────────────────────────────────────────── */
    /* prev_raw_*: last raw reading, used to compute per-frame deltas for unwrapping  */
    /* cont_*:     continuous (unwrapped) angle in degrees, zeroed at startup / R key */
    /* smooth_*:   low-pass filtered continuous angle fed to the renderer             */
    float prev_raw_roll  = 0.0f, prev_raw_pitch  = 0.0f, prev_raw_yaw  = 0.0f;
    float cont_roll      = 0.0f, cont_pitch      = 0.0f, cont_yaw      = 0.0f;
    float smooth_yaw_d   = 0.0f, smooth_pitch_d  = 0.0f, smooth_roll_d = 0.0f;
    int   zeroed = 0;

    /* ── Render loop ───────────────────────────────────────────────────────── */
    int   lighting_on = 1;
    int   l_key_prev  = GLFW_RELEASE;
    int   r_key_prev  = GLFW_RELEASE;
    float last_time   = (float)glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        float current_time = (float)glfwGetTime();
        float dt = current_time - last_time;
        last_time = current_time;
        if (dt <= 0.0f) dt = 0.001f;

        /* ── Keyboard ──────────────────────────────────────────────────────── */
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, 1);

        /* L — toggle shading */
        int l_key_curr = glfwGetKey(window, GLFW_KEY_L);
        if (l_key_curr == GLFW_PRESS && l_key_prev == GLFW_RELEASE) {
            lighting_on = !lighting_on;
            glfwSetWindowTitle(window, lighting_on
                ? "Pyramid — Direct IMU  [L] shading: ON"
                : "Pyramid — Direct IMU  [L] shading: OFF");
        }
        l_key_prev = l_key_curr;

        /* R — re-zero: reset continuous angles to 0 from current position */
        int r_key_curr = glfwGetKey(window, GLFW_KEY_R);
        if (r_key_curr == GLFW_PRESS && r_key_prev == GLFW_RELEASE) {
            cont_roll  = 0.0f;
            cont_pitch = 0.0f;
            cont_yaw   = 0.0f;
            smooth_yaw_d = smooth_pitch_d = smooth_roll_d = 0.0f;
            yaw = pitch = roll = 0.0f;
        }
        r_key_prev = r_key_curr;

        /* ── IMU: unwrap then smooth ─────────────────────────────────────────
           Each new reading computes a delta from the previous raw value.
           If the delta exceeds ±180° the angle wrapped — we correct it so
           the continuous angle never jumps.
           Smoothing uses an exponential low-pass with a fixed time constant
           (SMOOTH_TAU) so the response is the same at any frame rate.
        ────────────────────────────────────────────────────────────────────── */
        float raw_r, raw_p, raw_y;
        if (serial_read_angles(&raw_r, &raw_p, &raw_y)) {
            if (!zeroed) {
                /* On first reading, anchor all three axes here */
                prev_raw_roll  = raw_r;
                prev_raw_pitch = raw_p;
                prev_raw_yaw   = raw_y;
                cont_roll = cont_pitch = cont_yaw = 0.0f;
                zeroed = 1;
            } else {
                /* Unwrap: clamp the per-reading delta to ±180° */
                float dr = raw_r - prev_raw_roll;
                float dp = raw_p - prev_raw_pitch;
                float dy = raw_y - prev_raw_yaw;
                if (dr >  180.0f) dr -= 360.0f;
                if (dr < -180.0f) dr += 360.0f;
                if (dp >  180.0f) dp -= 360.0f;
                if (dp < -180.0f) dp += 360.0f;
                if (dy >  180.0f) dy -= 360.0f;
                if (dy < -180.0f) dy += 360.0f;
                cont_roll  += dr;
                cont_pitch += dp;
                cont_yaw   += dy;
            }
            prev_raw_roll  = raw_r;
            prev_raw_pitch = raw_p;
            prev_raw_yaw   = raw_y;
        }

        /* Smooth every frame toward the current continuous target */
        float alpha = expf(-dt / SMOOTH_TAU);
        smooth_yaw_d   = alpha * smooth_yaw_d   + (1.0f - alpha) * cont_roll;
        smooth_pitch_d = alpha * smooth_pitch_d + (1.0f - alpha) * cont_pitch;
        smooth_roll_d  = alpha * smooth_roll_d  + (1.0f - alpha) * cont_yaw;

        if (zeroed) {
            yaw   = smooth_yaw_d   * DEG_TO_RAD;
            pitch = smooth_pitch_d * DEG_TO_RAD;
            roll  = smooth_roll_d  * DEG_TO_RAD;
        }

        /* ── Build model matrix ────────────────────────────────────────────── */
        mat4 rot_x, rot_y, rot_z, tmp, model;
        mat4_rotate_x(rot_x, pitch);
        mat4_rotate_y(rot_y, yaw);
        mat4_rotate_z(rot_z, roll);
        mat4_multiply(tmp,   rot_y, rot_x);
        mat4_multiply(model, tmp,   rot_z);

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glUniformMatrix4fv(modelLoc,      1, GL_FALSE, model);
        glUniformMatrix4fv(viewLoc,       1, GL_FALSE, view);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection);
        glUniform1i(lightingEnabledLoc, lighting_on);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 18);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    /* ── Cleanup ───────────────────────────────────────────────────────────── */
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    if (serial_fd >= 0) close(serial_fd);

    glfwTerminate();
    return 0;
}
