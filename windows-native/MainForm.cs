using System;
using System.Drawing;
using System.Text;
using System.Text.Json;
using System.Windows.Forms;
using ChromaDBNavigator.Windows.Services;

namespace ChromaDBNavigator.Windows;

public sealed class MainForm : Form
{
    private readonly BackendClient _backend;

    private readonly TextBox _dbPathText = new() { ReadOnly = true, Dock = DockStyle.Fill };
    private readonly TextBox _collectionText = new() { Dock = DockStyle.Fill, Text = "references" };
    private readonly TextBox _folderText = new() { ReadOnly = true, Dock = DockStyle.Fill };
    private readonly Label _statusLabel = new() { Dock = DockStyle.Fill, Text = "Ready" };

    private readonly ListView _fileList = new() { Dock = DockStyle.Fill, View = View.List };
    private readonly TextBox _fileInfoText = new() { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical };

    private readonly TextBox _searchInput = new() { Dock = DockStyle.Fill };
    private readonly NumericUpDown _searchCount = new() { Dock = DockStyle.Right, Width = 90, Minimum = 1, Maximum = 50, Value = 10 };
    private readonly ListBox _searchResults = new() { Dock = DockStyle.Fill };
    private readonly TextBox _searchDetail = new() { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical };

    private readonly ListBox _documentList = new() { Dock = DockStyle.Fill };
    private readonly TextBox _documentDetail = new() { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical };

    private readonly TextBox _statsText = new() { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical };

    private string _currentFolder = string.Empty;

    public MainForm()
    {
        Text = "ChromaDB Navigator (Windows Native)";
        Width = 1400;
        Height = 900;
        StartPosition = FormStartPosition.CenterScreen;

        var repoRoot = AppContext.BaseDirectory;
        while (!File.Exists(Path.Combine(repoRoot, "native_backend_server.py")) && Directory.GetParent(repoRoot) is not null)
        {
            repoRoot = Directory.GetParent(repoRoot)!.FullName;
        }

        var pythonExe = Environment.GetEnvironmentVariable("CHROMA_PYTHON_EXE");
        if (string.IsNullOrWhiteSpace(pythonExe))
        {
            pythonExe = "python";
        }

        _backend = new BackendClient(repoRoot, pythonExe);

        BuildLayout();
        WireEvents();

        Shown += async (_, _) =>
        {
            try
            {
                await InitializeBackendAsync();
            }
            catch (Exception ex)
            {
                ShowError(ex.Message);
            }
        };

        FormClosed += (_, _) => _backend.Dispose();
    }

    private void BuildLayout()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
            ColumnCount = 1,
        };
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        root.Controls.Add(BuildTopPanel(), 0, 0);
        root.Controls.Add(BuildMainPanel(), 0, 1);
        root.Controls.Add(BuildStatusPanel(), 0, 2);

        Controls.Add(root);
    }

    private Control BuildTopPanel()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            RowCount = 3,
            ColumnCount = 5,
            Padding = new Padding(10),
        };

        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

        panel.Controls.Add(new Label { Text = "Database Path:", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 0);
        panel.Controls.Add(_dbPathText, 1, 0);

        var browseDbButton = new Button { Text = "Browse DB" };
        browseDbButton.Click += async (_, _) => await BrowseDbPathAsync();
        panel.Controls.Add(browseDbButton, 2, 0);

        panel.Controls.Add(new Label { Text = "Collection:", AutoSize = true, Anchor = AnchorStyles.Left }, 3, 0);
        panel.Controls.Add(_collectionText, 4, 0);

        panel.Controls.Add(new Label { Text = "PDF Folder:", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 1);
        panel.Controls.Add(_folderText, 1, 1);

        var browseFolderButton = new Button { Text = "Browse Folder" };
        browseFolderButton.Click += (_, _) => BrowseFolder();
        panel.Controls.Add(browseFolderButton, 2, 1);

        var applyCollectionButton = new Button { Text = "Apply Collection" };
        applyCollectionButton.Click += async (_, _) => await InitializeBackendAsync();
        panel.Controls.Add(applyCollectionButton, 3, 1);

        var syncButton = new Button { Text = "Sync Database", Name = "SyncButton" };
        syncButton.Click += async (_, _) => await SyncDatabaseAsync(syncButton);
        panel.Controls.Add(syncButton, 4, 1);

        return panel;
    }

    private Control BuildMainPanel()
    {
        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterDistance = 360,
        };

        split.Panel1.Controls.Add(BuildFilePanel());
        split.Panel2.Controls.Add(BuildTabPanel());

        return split;
    }

    private Control BuildFilePanel()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
            ColumnCount = 1,
            Padding = new Padding(8),
        };

        panel.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 60));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 40));

        panel.Controls.Add(new Label { Text = "PDF Files", AutoSize = true, Font = new Font(Font, FontStyle.Bold) }, 0, 0);
        panel.Controls.Add(_fileList, 0, 1);
        panel.Controls.Add(_fileInfoText, 0, 2);

        return panel;
    }

    private Control BuildTabPanel()
    {
        var tabs = new TabControl { Dock = DockStyle.Fill };
        tabs.TabPages.Add(BuildSearchTab());
        tabs.TabPages.Add(BuildBrowseTab());
        tabs.TabPages.Add(BuildStatsTab());
        return tabs;
    }

    private TabPage BuildSearchTab()
    {
        var tab = new TabPage("Search");
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
            ColumnCount = 1,
            Padding = new Padding(8),
        };

        panel.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 50));

        var controls = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true };
        var searchButton = new Button { Text = "Search" };
        searchButton.Click += async (_, _) => await SearchAsync();

        controls.Controls.Add(new Label { Text = "Query:", AutoSize = true, Padding = new Padding(0, 8, 0, 0) });
        _searchInput.Width = 500;
        controls.Controls.Add(_searchInput);
        controls.Controls.Add(searchButton);
        controls.Controls.Add(new Label { Text = "Results:", AutoSize = true, Padding = new Padding(10, 8, 0, 0) });
        controls.Controls.Add(_searchCount);

        panel.Controls.Add(controls, 0, 0);
        panel.Controls.Add(_searchResults, 0, 1);
        panel.Controls.Add(_searchDetail, 0, 2);

        tab.Controls.Add(panel);
        return tab;
    }

    private TabPage BuildBrowseTab()
    {
        var tab = new TabPage("Browse");
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
            ColumnCount = 1,
            Padding = new Padding(8),
        };

        panel.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 50));

        var refreshButton = new Button { Text = "Refresh Documents" };
        refreshButton.Click += async (_, _) => await RefreshDocumentsAsync();

        panel.Controls.Add(refreshButton, 0, 0);
        panel.Controls.Add(_documentList, 0, 1);
        panel.Controls.Add(_documentDetail, 0, 2);

        tab.Controls.Add(panel);
        return tab;
    }

    private TabPage BuildStatsTab()
    {
        var tab = new TabPage("Statistics");
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 2,
            ColumnCount = 1,
            Padding = new Padding(8),
        };

        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        panel.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        var refreshButton = new Button { Text = "Refresh Statistics" };
        refreshButton.Click += async (_, _) => await RefreshStatsAsync();

        panel.Controls.Add(_statsText, 0, 0);
        panel.Controls.Add(refreshButton, 0, 1);

        tab.Controls.Add(panel);
        return tab;
    }

    private Control BuildStatusPanel()
    {
        var panel = new Panel { Dock = DockStyle.Bottom, Height = 30, Padding = new Padding(8, 4, 8, 4) };
        panel.Controls.Add(_statusLabel);
        return panel;
    }

    private void WireEvents()
    {
        _fileList.SelectedIndexChanged += (_, _) => ShowSelectedFileInfo();
        _searchResults.SelectedIndexChanged += (_, _) => ShowSelectedSearchResult();
        _documentList.SelectedIndexChanged += (_, _) => ShowSelectedDocument();
        _searchInput.KeyDown += async (_, e) =>
        {
            if (e.KeyCode == Keys.Enter)
            {
                e.Handled = true;
                e.SuppressKeyPress = true;
                await SearchAsync();
            }
        };
    }

    private async Task InitializeBackendAsync()
    {
        SetStatus("Initializing backend...");
        var response = await _backend.InitializeAsync(_dbPathText.Text, _collectionText.Text.Trim());
        var collection = response.GetProperty("collection").GetString();
        var dbPath = response.GetProperty("db_path").GetString();
        _collectionText.Text = collection ?? "references";
        _dbPathText.Text = dbPath ?? string.Empty;
        SetStatus($"Ready (collection: {_collectionText.Text})");
    }

    private async Task BrowseDbPathAsync()
    {
        using var dialog = new FolderBrowserDialog();
        if (dialog.ShowDialog() != DialogResult.OK)
        {
            return;
        }

        _dbPathText.Text = dialog.SelectedPath;
        await InitializeBackendAsync();
    }

    private void BrowseFolder()
    {
        using var dialog = new FolderBrowserDialog();
        if (dialog.ShowDialog() != DialogResult.OK)
        {
            return;
        }

        _currentFolder = dialog.SelectedPath;
        _folderText.Text = _currentFolder;
        LoadFilesFromFolder();
        SetStatus($"Selected folder: {_currentFolder}");
    }

    private void LoadFilesFromFolder()
    {
        _fileList.Items.Clear();
        _fileInfoText.Clear();

        if (string.IsNullOrWhiteSpace(_currentFolder) || !Directory.Exists(_currentFolder))
        {
            return;
        }

        foreach (var file in Directory.GetFiles(_currentFolder, "*.pdf").OrderBy(Path.GetFileName))
        {
            _fileList.Items.Add(new ListViewItem(file));
        }
    }

    private void ShowSelectedFileInfo()
    {
        if (_fileList.SelectedItems.Count == 0)
        {
            return;
        }
        var filePath = _fileList.SelectedItems[0].Text;

        try
        {
            var info = new FileInfo(filePath);
            var builder = new StringBuilder();
            builder.AppendLine($"Filename: {info.Name}");
            builder.AppendLine($"Path: {info.FullName}");
            builder.AppendLine($"Size: {info.Length:N0} bytes");
            _fileInfoText.Text = builder.ToString();
        }
        catch (Exception ex)
        {
            _fileInfoText.Text = ex.Message;
        }
    }

    private async Task SyncDatabaseAsync(Button syncButton)
    {
        if (string.IsNullOrWhiteSpace(_currentFolder) || !Directory.Exists(_currentFolder))
        {
            ShowError("Select a valid PDF folder first.");
            return;
        }

        syncButton.Enabled = false;
        try
        {
            SetStatus("Previewing sync changes...");
            var preview = await _backend.PreviewSyncAsync(_currentFolder);
            var summary = BuildSyncPreviewMessage(preview);

            var confirm = MessageBox.Show(
                summary + "\n\nContinue with synchronization?",
                "Confirm Synchronization",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question,
                MessageBoxDefaultButton.Button2);

            if (confirm != DialogResult.Yes)
            {
                return;
            }

            SetStatus("Synchronizing database...");
            var result = await _backend.SyncAsync(_currentFolder);

            var added = result.GetProperty("added").GetInt32();
            var removed = result.GetProperty("removed").GetInt32();
            var corruptedCount = result.GetProperty("corrupted").GetArrayLength();

            MessageBox.Show(
                $"Sync completed.\n\nFiles processed: {added}\nFiles removed: {removed}\nFiles with errors: {corruptedCount}",
                "Sync Complete",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information);

            await RefreshDocumentsAsync();
            await RefreshStatsAsync();
            SetStatus("Sync completed");
        }
        catch (Exception ex)
        {
            ShowError(ex.Message);
        }
        finally
        {
            syncButton.Enabled = true;
        }
    }

    private string BuildSyncPreviewMessage(JsonElement preview)
    {
        var newFiles = preview.GetProperty("new_files").GetArrayLength();
        var modifiedFiles = preview.GetProperty("modified_files").GetArrayLength();
        var removedFiles = preview.GetProperty("removed_files").GetArrayLength();
        var corruptedFiles = preview.GetProperty("corrupted_files").GetArrayLength();

        if (newFiles == 0 && modifiedFiles == 0 && removedFiles == 0 && corruptedFiles == 0)
        {
            return "Database is already up to date.";
        }

        return $"Changes detected:\n- New: {newFiles}\n- Modified: {modifiedFiles}\n- Removed: {removedFiles}\n- Corrupted: {corruptedFiles}";
    }

    private async Task SearchAsync()
    {
        var query = _searchInput.Text.Trim();
        if (string.IsNullOrWhiteSpace(query))
        {
            return;
        }

        try
        {
            SetStatus("Searching...");
            var result = await _backend.SearchAsync(query, (int)_searchCount.Value);
            var items = result.GetProperty("items");

            _searchResults.Items.Clear();
            _searchDetail.Clear();

            foreach (var item in items.EnumerateArray())
            {
                var meta = item.GetProperty("metadata");
                var filename = meta.TryGetProperty("filename", out var f) ? f.GetString() : "Unknown";
                var page = meta.TryGetProperty("page_number", out var p) ? p.ToString() : "?";
                var chunkType = meta.TryGetProperty("chunk_type", out var c) ? c.GetString() : "unknown";

                var label = $"{filename} (Page {page}, {chunkType})";
                _searchResults.Items.Add(new UiItem(label, item.Clone()));
            }

            SetStatus($"Found {_searchResults.Items.Count} results");
        }
        catch (Exception ex)
        {
            ShowError(ex.Message);
        }
    }

    private void ShowSelectedSearchResult()
    {
        if (_searchResults.SelectedItem is not UiItem item)
        {
            return;
        }

        var result = item.Payload;
        var metadata = result.GetProperty("metadata");
        var text = result.GetProperty("text").GetString();

        var builder = new StringBuilder();
        builder.AppendLine($"Filename: {GetString(metadata, "filename")}");
        builder.AppendLine($"Page: {GetString(metadata, "page_number")}");
        builder.AppendLine($"Chunk Type: {GetString(metadata, "chunk_type")}");
        builder.AppendLine($"APA Reference: {GetString(metadata, "apa_reference")}");
        builder.AppendLine($"Chunk Size: {GetString(metadata, "chunk_size")}");
        builder.AppendLine($"Extraction Date: {GetString(metadata, "extraction_date")}");
        builder.AppendLine();
        builder.AppendLine("--- Content ---");
        builder.AppendLine(text);

        _searchDetail.Text = builder.ToString();
    }

    private async Task RefreshDocumentsAsync()
    {
        try
        {
            SetStatus("Loading documents...");
            var result = await _backend.ListDocumentsAsync();

            _documentList.Items.Clear();
            _documentDetail.Clear();

            foreach (var doc in result.GetProperty("items").EnumerateArray())
            {
                var meta = doc.GetProperty("metadata");
                var filename = GetString(meta, "filename");
                var page = GetString(meta, "page_number");
                var chunkType = GetString(meta, "chunk_type");
                var label = $"{filename} (Page {page}, {chunkType})";
                _documentList.Items.Add(new UiItem(label, doc.Clone()));
            }

            SetStatus($"Loaded {_documentList.Items.Count} chunks");
        }
        catch (Exception ex)
        {
            ShowError(ex.Message);
        }
    }

    private void ShowSelectedDocument()
    {
        if (_documentList.SelectedItem is not UiItem item)
        {
            return;
        }

        var doc = item.Payload;
        var metadata = doc.GetProperty("metadata");
        var text = doc.GetProperty("text").GetString();

        var builder = new StringBuilder();
        builder.AppendLine($"Filename: {GetString(metadata, "filename")}");
        builder.AppendLine($"Page: {GetString(metadata, "page_number")}");
        builder.AppendLine($"Chunk Type: {GetString(metadata, "chunk_type")}");
        builder.AppendLine($"APA Reference: {GetString(metadata, "apa_reference")}");
        builder.AppendLine($"Chunk Size: {GetString(metadata, "chunk_size")}");
        builder.AppendLine($"Total Pages: {GetString(metadata, "total_pages")}");
        builder.AppendLine($"File Size: {GetString(metadata, "file_size")}");
        builder.AppendLine($"Extraction Date: {GetString(metadata, "extraction_date")}");
        builder.AppendLine();
        builder.AppendLine("--- Content ---");
        builder.AppendLine(text);

        _documentDetail.Text = builder.ToString();
    }

    private async Task RefreshStatsAsync()
    {
        try
        {
            SetStatus("Loading statistics...");
            var stats = await _backend.StatsAsync();

            var builder = new StringBuilder();
            builder.AppendLine("Database Statistics");
            builder.AppendLine("==================");
            builder.AppendLine();
            builder.AppendLine($"Total Chunks: {GetString(stats, "total_chunks")}");
            builder.AppendLine($"Unique Files: {GetString(stats, "unique_files")}");
            builder.AppendLine();

            if (stats.TryGetProperty("device_info", out var device))
            {
                builder.AppendLine("Embedding Device:");
                builder.AppendLine($"  Device: {GetString(device, "device")}");
                builder.AppendLine($"  Description: {GetString(device, "device_name")}");
                if (device.TryGetProperty("note", out var note))
                {
                    builder.AppendLine($"  Note: {note.GetString()}");
                }
                builder.AppendLine();
            }

            if (stats.TryGetProperty("chunk_types", out var chunkTypes) && chunkTypes.ValueKind == JsonValueKind.Object)
            {
                builder.AppendLine("Chunk Types:");
                foreach (var kv in chunkTypes.EnumerateObject())
                {
                    builder.AppendLine($"  {kv.Name}: {kv.Value}");
                }
                builder.AppendLine();
            }

            if (stats.TryGetProperty("file_extensions", out var ext) && ext.ValueKind == JsonValueKind.Object)
            {
                builder.AppendLine("File Extensions:");
                foreach (var kv in ext.EnumerateObject())
                {
                    builder.AppendLine($"  {kv.Name}: {kv.Value}");
                }
            }

            _statsText.Text = builder.ToString();
            SetStatus("Statistics updated");
        }
        catch (Exception ex)
        {
            ShowError(ex.Message);
        }
    }

    private static string GetString(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var value))
        {
            return "N/A";
        }

        return value.ValueKind switch
        {
            JsonValueKind.Null => "N/A",
            JsonValueKind.String => value.GetString() ?? "",
            _ => value.ToString(),
        };
    }

    private void SetStatus(string text)
    {
        _statusLabel.Text = text;
    }

    private void ShowError(string message)
    {
        SetStatus("Error");
        MessageBox.Show(message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    private sealed class UiItem
    {
        public UiItem(string label, JsonElement payload)
        {
            Label = label;
            Payload = payload;
        }

        public string Label { get; }
        public JsonElement Payload { get; }

        public override string ToString() => Label;
    }
}
