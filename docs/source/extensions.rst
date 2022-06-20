.. _extensions:

Environment extensions
======================

The environment extension is our fuctionality that allows agent to infer latent observations that are
not originally supported by the environment. You can either choose one of our built-in extensions or
implemet your own with the help of this short guide.

.. _custom_exts:

Custom extensions
-----------------

The main axis of this module is :ref:`abstract class <base_ext>` ``BaseExt``, which provides an
interface for all of the other environments, both the in-built ones and those implemented by the user.

.. _base_ext:

BaseExt
-------

.. currentmodule:: reinforced_lib.exts.base_ext

.. autoclass:: BaseExt
    :members:

List of extensions
------------------

IEEE 802.11ax
~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.exts.ieee_802_11_ax

.. autoclass:: IEEE_802_11_ax
    :show-inheritance:
    :members:
