const std = @import("std");
const builtin = @import("builtin");
const w = std.os.windows;

// locale C
extern "c" fn setlocale(category: c_int, locale: ?[*:0]const u8) ?[*:0]const u8;
pub const LC_ALL: c_int = 6;

// WinAPI
pub extern "kernel32" fn SetConsoleOutputCP(wCodePageID: w.UINT) callconv(.winapi) w.BOOL;
pub extern "kernel32" fn SetConsoleCP(wCodePageID: w.UINT) callconv(.winapi) w.BOOL;

pub fn setUtf8Console() void {
    if (builtin.target.os.tag == .windows) {
        _ = SetConsoleOutputCP(65001); // UTF-8
        _ = SetConsoleCP(65001);
        _ = setlocale(LC_ALL, "fr-FR.UTF-8");
        // std.debug.print("éè☀️ç?🌳$£*µ☕\n", .{});
    }
}
