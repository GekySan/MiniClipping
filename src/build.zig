const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    // Vérification de la version de Zig
    const required_version = std.SemanticVersion.parse("0.15.1") catch unreachable;
    if (builtin.zig_version.order(required_version) != .eq) {
        std.debug.panic(
            \\Ce projet doit être compilé exactement avec Zig {any}.
            \\Version actuellement utilisée : {any}
        , .{ required_version, builtin.zig_version });
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("clipping.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Lien avec libc (-lc)
    exe.linkSystemLibrary("c");

    b.installArtifact(exe);
}
