from .anima_artist_encode import AnimaArtistEncode

NODE_CLASS_MAPPINGS = {
	"AnimaArtistEncode": AnimaArtistEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"AnimaArtistEncode": "Anima Artist Encode",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]