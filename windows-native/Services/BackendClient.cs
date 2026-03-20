using System;
using System.Diagnostics;
using System.Text.Json;

namespace ChromaDBNavigator.Windows.Services;

public sealed class BackendClient : IDisposable
{
    private readonly Process _process;
    private readonly object _ioLock = new();
    private int _nextId = 1;
    private bool _disposed;

    public BackendClient(string workingDirectory, string pythonExe)
    {
        var scriptPath = Path.Combine(workingDirectory, "native_backend_server.py");
        if (!File.Exists(scriptPath))
        {
            throw new FileNotFoundException("Could not find backend server script", scriptPath);
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = $"-u \"{scriptPath}\"",
            WorkingDirectory = workingDirectory,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        _process = new Process { StartInfo = startInfo };
        _process.Start();

        // Drain stderr so the backend never blocks on logging output.
        _ = Task.Run(async () =>
        {
            while (!_process.HasExited)
            {
                var line = await _process.StandardError.ReadLineAsync().ConfigureAwait(false);
                if (line is null)
                {
                    break;
                }
                Debug.WriteLine($"[PY] {line}");
            }
        });
    }

    public Task<JsonElement> InitializeAsync(string? dbPath, string collection)
        => SendAsync("initialize", new Dictionary<string, object?>
        {
            ["db_path"] = string.IsNullOrWhiteSpace(dbPath) ? null : dbPath,
            ["collection"] = collection,
        });

    public Task<JsonElement> PreviewSyncAsync(string folder)
        => SendAsync("preview_sync", new Dictionary<string, object?> { ["folder"] = folder });

    public Task<JsonElement> SyncAsync(string folder)
        => SendAsync("sync", new Dictionary<string, object?> { ["folder"] = folder });

    public Task<JsonElement> SearchAsync(string query, int nResults)
        => SendAsync("search", new Dictionary<string, object?>
        {
            ["query"] = query,
            ["n_results"] = nResults,
        });

    public Task<JsonElement> ListDocumentsAsync()
        => SendAsync("list_documents", new Dictionary<string, object?>());

    public Task<JsonElement> StatsAsync()
        => SendAsync("stats", new Dictionary<string, object?>());

    private Task<JsonElement> SendAsync(string action, Dictionary<string, object?> payload)
    {
        return Task.Run(() =>
        {
            lock (_ioLock)
            {
                ThrowIfDisposed();
                if (_process.HasExited)
                {
                    throw new InvalidOperationException("Python backend is not running.");
                }

                var id = _nextId++;
                var request = JsonSerializer.Serialize(new
                {
                    id,
                    action,
                    payload,
                });

                _process.StandardInput.WriteLine(request);
                _process.StandardInput.Flush();

                var line = _process.StandardOutput.ReadLine();
                if (string.IsNullOrWhiteSpace(line))
                {
                    throw new InvalidOperationException("No response from backend.");
                }

                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;

                var responseId = root.GetProperty("id").GetInt32();
                if (responseId != id)
                {
                    throw new InvalidOperationException($"Backend response id mismatch. Expected {id}, got {responseId}.");
                }

                var ok = root.GetProperty("ok").GetBoolean();
                if (!ok)
                {
                    var error = root.TryGetProperty("error", out var errorProp)
                        ? errorProp.GetString()
                        : "Unknown backend error";
                    throw new InvalidOperationException(error);
                }

                return root.GetProperty("result").Clone();
            }
        });
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(BackendClient));
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        try
        {
            if (!_process.HasExited)
            {
                _process.StandardInput.Close();
                _process.Kill(entireProcessTree: true);
            }
        }
        catch
        {
            // Best effort shutdown.
        }
        finally
        {
            _process.Dispose();
        }
    }
}
