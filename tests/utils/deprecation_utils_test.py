import pytest
import warnings
from asteroid.utils import deprecation_utils as dp


def test_warning():
    with pytest.warns(dp.VisibleDeprecationWarning):
        warnings.warn("Expected warning.", dp.VisibleDeprecationWarning)


def test_deprecated():
    class Foo:
        def new_func(self):
            pass

        @dp.mark_deprecated("Please use `new_func`", "0.5.0")
        def old_func(self):
            pass

        @dp.mark_deprecated("Please use `new_func`")
        def no_version_old_func(self):
            pass

        @dp.mark_deprecated(message="")
        def no_message_old_func(self):
            pass

    foo = Foo()
    foo.new_func()

    with pytest.warns(dp.VisibleDeprecationWarning) as record:
        foo.old_func()
    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert "0.5.0" in record[0].message.args[0]

    with pytest.warns(dp.VisibleDeprecationWarning):
        foo.no_version_old_func()
        foo.no_message_old_func()


def test_is_overidden():
    class Foo:
        def some_func(self):
            return None

    class Bar(Foo):
        def some_func(self):
            something_changed = None
            return None

    class Ho(Bar):
        pass

    # On class
    assert dp.is_overridden("some_func", Bar, parent=Foo)
    assert dp.is_overridden("some_func", Bar)
    # On instance
    bar = Bar()
    assert dp.is_overridden("some_func", bar, parent=Foo)
    assert dp.is_overridden("some_func", bar)

    class Hey(Foo):
        def some_other_func(self):
            return None

    # On class
    assert not dp.is_overridden("some_func", Hey, parent=Foo)
    # On instance
    hey = Hey()
    assert not dp.is_overridden("some_func", hey, parent=Foo)
    assert not dp.is_overridden("some_func", hey, parent=Foo)

    with pytest.raises(RuntimeError):
        dp.is_overridden("some_func", hey, parent=Bar)

    with pytest.raises(RuntimeError):
        dp.is_overridden("some_other_func", hey, parent=Foo)

    with pytest.raises(RuntimeError):
        dp.is_overridden("some_other_func", hey)
