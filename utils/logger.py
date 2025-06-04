from tensorboardX import SummaryWriter

class Logger:
    """
    Wrapper class for TensorBoard logging with optional enable/disable functionality.
    """
    
    def __init__(self, enable_logging=True, log_dir=None):
        """
        Initialize logger with optional TensorBoard integration.
        
        Args:
            enable_logging (bool, optional): Whether to enable TensorBoard logging. Defaults to True.
            log_dir (str, optional): Directory for TensorBoard logs. If None, uses default.
        
        Returns:
            None: Initializes logger instance
        """
        self.enable_logging = enable_logging
        self.writer = SummaryWriter(log_dir=log_dir) if enable_logging else None
    
    def add_figure(self, *args, **kwargs):
        """
        Add figure to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs figure to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_figure(*args, **kwargs)
    
    def add_text(self, *args, **kwargs):
        """
        Add text to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs text to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_text(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        """
        Add video to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs video to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_video(*args, **kwargs)
    
    def add_scalar(self, *args, **kwargs):
        """
        Add scalar to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs scalar to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_scalar(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        """
        Add histogram to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs histogram to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_histogram(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        """
        Add image to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs image to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_image(*args, **kwargs)
    
    def add_graph(self, *args, **kwargs):
        """
        Add graph to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs graph to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_graph(*args, **kwargs)

    def add_audio(self, *args, **kwargs):
        """
        Add audio to TensorBoard if logging is enabled.
        
        Args:
            *args: Positional arguments passed to TensorBoard writer
            **kwargs: Keyword arguments passed to TensorBoard writer
        
        Returns:
            None: Logs audio to TensorBoard or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.add_audio(*args, **kwargs)

    def flush(self):
        """
        Flush TensorBoard writer if logging is enabled.
        
        Args:
            None
        
        Returns:
            None: Flushes TensorBoard writer or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.flush()

    def close(self):
        """
        Close TensorBoard writer if logging is enabled.
        
        Args:
            None
        
        Returns:
            None: Closes TensorBoard writer or does nothing if disabled
        """
        if self.enable_logging:
            self.writer.close()