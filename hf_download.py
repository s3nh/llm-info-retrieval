from huggingface_hub import snapshot_download
if __name__=='__main__':
  repo_id = "PygmalionAI/mythalion-13b"
  outpath = 'model1'
  os.makedirs(outpath, exist_ok=True)
  # Select branch
  revision="main"
  # Download model
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id=repo_id,
                    allow_patterns = '*',
                    local_dir= outpath,
                    ignore_patterns= '*.bin',
                    local_dir_use_symlinks=False)
